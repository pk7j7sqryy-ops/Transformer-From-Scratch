"""
    训练循环
    数据流: DataLoader → Batch → model → loss → backward → step
"""
import time
from torch import optim
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config.config import *
from proprocess.dataset import get_dataloader
from model.model import build_model
from runner.loss import LabelSmoothing
from runner.masks import make_padding_mask, make_decoder_mask
from torch.optim.lr_scheduler import LambdaLR

class Batch:
    def __init__(self, src, tgt, pad_id=0):
        '''
        #   tgt_y = tgt[:, 1:]     ← 右移一位，作为标签（预测下一个词）
        #   tgt_mask = padding + causal mask
        #   ntokens = 非PAD token 数
        :param src:
        :param tgt:
        :param pad_id:
        '''
        self.src = src
        self.src_mask = make_padding_mask(src, pad_id)
        self.tgt = tgt[:, :-1]
        self.tgt_y = tgt[:, 1:]
        self.tgt_mask = make_decoder_mask(self.tgt, pad_id)
        self.ntokens = (self.tgt_y != pad_id).sum()

def train():
    #1、定义基本信息
    config = TransformerConfig()

    #2、构建模型
    model=build_model(config)


    #3、加载数据
    train_loader = get_dataloader('train',config.batch_size,config.pad_id, limit=config.limit)
    val_loader = get_dataloader('valid',config.batch_size,config.pad_id, limit=config.limit)

    #4、loss
    loss_fn=LabelSmoothing(
        size=config.tgt_vocab_size,
        padding_idx=config.pad_id,
        smoothing=0.1
    )
    #5、训练器
    trainer=Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=config
    )
    #7、训练
    trainer.train()

class Trainer():
    #1、初始化
    def __init__(self,model, train_loader, val_loader, loss_fn,config):
        # 定义属性
        # 训练配置
        self.config = config
        # 模型设备
        self.model = model.to(self.config.device)
        self.device = self.config.device
        # 数据集及函数
        self.train_loader = train_loader
        self.val_loader = val_loader
        #定义损失函数
        self.loss_fn=loss_fn
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.lr,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        # 日志写入器
        self.writer = SummaryWriter(log_dir=str(LOGS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S")))
        # AMP缩放器
        self.scaler = torch.amp.GradScaler(device=self.config.device, enabled=self.config.use_amp)

        # 全局变量
        self.early_stopping_best_score = float("inf")  # 模型最佳得分
        self.early_stopping_counter = 0  # 容忍度计数器
        self.step = 0
        self.start_epoch = 0
        # 定义保存目录
        self.best_model_path = MODELS_DIR / BEST_MODEL
        self.last_ckpt_path = MODELS_DIR / 'last' / 'checkpoint.pt'

        self.scheduler = LambdaLR(self.optimizer, self.lr_lambda)

    def lr_lambda(self,step):
        step += 1
        if step < self.config.warmup_steps:
            # 线性增长：0 → 1（lr 从 0 爬到 base_lr）
            return step / self.config.warmup_steps
        else:
            # 衰减：1 → 0（lr 从 base_lr 慢慢降）
            return self.config.warmup_steps ** 0.5 / step ** 0.5
    #内部方法：传入数据集 获取数据加载起

    #内部方法：一次迭代训练 返回损失
    def _train_one_step(self,batch_data):
        self.model.train()
        #0.梯度清0
        self.optimizer.zero_grad(set_to_none=True)
        # 1. 用 Batch 类加工数据: {'src': [B,S], 'tgt': [B,T]} → src, tgt, tgt_y, src_mask, tgt_mask
        batch = Batch(
            batch_data['src'].to(self.device),
            batch_data['tgt'].to(self.device),
            pad_id=self.config.pad_id,
        )
        with torch.autocast(
                device_type=self.config.device,
                dtype=torch.float16,
                enabled=self.config.use_amp,
        ):
        #2、前向传播,传入中文、中文填充、英文、英文填充和因果填充
            out=self.model(batch.src,batch.tgt,batch.src_mask,batch.tgt_mask)
            out=self.model.generator(out)

            #3、计算损失
            loss_sum = self.loss_fn(
                out.reshape(-1,out.size(-1)),
                batch.tgt_y.reshape(-1),
            )
            ntokens = batch.ntokens.clamp_min(1)
            loss = loss_sum / ntokens
        #4、反向传播
        self.scaler.scale(loss).backward()

        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )
        #5、更新参数
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()#lr 会自动从 0 增长到峰值，再慢慢衰减

        return (
            loss_sum.detach().item(),
            batch.ntokens.item(),
            grad_norm.item(),
        )
    #内部方法：传入当前验证指标，实现早停逻辑 返回bool值
    def _should_early_stop(self,val_loss):
        if val_loss<self.early_stopping_best_score:
            self.early_stopping_best_score=val_loss
            self.early_stopping_counter=0
            torch.save(self.model.state_dict(),self.best_model_path)
            tqdm.write('模型保存成功！')
            return False
        else:
            self.early_stopping_counter+=1
            if self.early_stopping_counter>=self.config.early_stopping_patience:
                return True
            else:
                return False
    #内部方法：保存检查点
    def _save_checkpoint(self, epoch):
        self.last_ckpt_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'step': self.step,
            'early_stopping_best_score': (
                self.early_stopping_best_score
            ),
            'early_stopping_counter': (
                self.early_stopping_counter
            ),
        }

        torch.save(checkpoint, self.last_ckpt_path)
        tqdm.write('检查点保存成功！')
    # def _save_checkpoint(self):
    #     """保存检查点"""
    #     self.last_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    #     checkpoint = {
    #         'model_state_dict': self.model.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #         'step': self.step,
    #         'early_stopping_best_score': self.early_stopping_best_score,
    #         'early_stopping_counter': self.early_stopping_counter,
    #         'scheduler_state_dict': self.scheduler.state_dict(),
    #     }
    #     torch.save(checkpoint, self.last_ckpt_path)
    #     tqdm.write("检查点保存成功！")
    #内部方法：加载检查点
    def _load_checkpoint(self):
        if not self.config.resume:
            print('resume=False，从头开始训练...')
            return

        if not self.last_ckpt_path.exists():
            print('没有发现检查点，从头开始训练...')
            return

        print('发现检查点，加载后继续训练...')

        checkpoint = torch.load(
            self.last_ckpt_path,
            map_location=self.device,
            weights_only=False,
        )

        self.model.load_state_dict(
            checkpoint['model_state_dict']
        )
        self.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict']
        )
        self.scheduler.load_state_dict(
            checkpoint['scheduler_state_dict']
        )

        self.step = checkpoint.get('step', 0)
        self.start_epoch = checkpoint.get('epoch', -1) + 1

        self.early_stopping_best_score = checkpoint.get(
            'early_stopping_best_score',
            float('inf'),
        )
        self.early_stopping_counter = checkpoint.get(
            'early_stopping_counter',
            0,
        )
    # def _load_checkpoint(self):
    #     """加载检查点"""
    #     if self.last_ckpt_path.exists():
    #         print("发现检查点，加载后继续训练...")
    #         checkpoint = torch.load(self.last_ckpt_path, weights_only=False)
    #         self.model.load_state_dict(checkpoint['model_state_dict'])
    #         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #         self.step = checkpoint['step']
    #         self.early_stopping_best_score = checkpoint['early_stopping_best_score']
    #         self.early_stopping_counter = checkpoint['early_stopping_counter']
    #         self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #     else:
    #         print("没有发现检查点，从头开始训练...")
    #训练
    def train(self):
        self._load_checkpoint()
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        interval_loss_sum = 0.0
        interval_tokens = 0

        try:
            for epoch in range(self.start_epoch, self.config.epoch):
                progress = tqdm(
                    self.train_loader,
                    desc=f'[Epoch {epoch + 1}]',
                )

                for batch_data in progress:
                    loss_sum, ntokens, grad_norm = self._train_one_step(
                        batch_data
                    )

                    self.step += 1
                    interval_loss_sum += loss_sum
                    interval_tokens += ntokens

                    if self.step % self.config.log_steps == 0:
                        train_loss = (
                                interval_loss_sum
                                / max(1, interval_tokens)
                        )
                        current_lr = self.optimizer.param_groups[0]['lr']

                        tqdm.write(
                            f'[Epoch {epoch + 1}] '
                            f'Step: {self.step} '
                            f'Train_Loss: {train_loss:.4f} '
                            f'LR: {current_lr:.8f} '
                            f'Grad_Norm: {grad_norm:.4f}'
                        )

                        self.writer.add_scalar(
                            'train/loss',
                            train_loss,
                            self.step,
                        )
                        self.writer.add_scalar(
                            'train/lr',
                            current_lr,
                            self.step,
                        )
                        self.writer.add_scalar(
                            'train/grad_norm',
                            grad_norm,
                            self.step,
                        )

                        interval_loss_sum = 0.0
                        interval_tokens = 0

                # 每个 epoch 结束后只验证一次
                metrics = self.evaluate()
                val_loss = metrics['loss']

                tqdm.write(
                    f'[Epoch {epoch + 1}] '
                    f'Validation_Loss: {val_loss:.4f}'
                )

                self.writer.add_scalar(
                    'val/loss',
                    val_loss,
                    self.step,
                )

                should_stop = self._should_early_stop(val_loss)

                # 每个 epoch 保存一次断点
                self._save_checkpoint(epoch)

                if should_stop:
                    tqdm.write('早停！')
                    break

        finally:
            self.writer.close()
    # def train(self):
    #     #1加载检查点，如果上次训练中断了，从检查点恢复（模型参数、优化器、步数）
    #     self._load_checkpoint()
    #     MODELS_DIR.mkdir(parents=True, exist_ok=True)
    #     interval_loss_sum = 0.0
    #     interval_tokens = 0
    #     #轮次训练
    #     for epoch in range(self.config.epoch):
    #         for batch_data in tqdm(self.train_loader,desc=f'[Epoch {epoch+1}]'):
    #
    #             ## 拿一个 batch → 前向 → loss → 反向 → 更新参数 → 返回 loss
    #             this_loss =self._train_one_step(batch_data)
    #
    #             # 判断是否要记录损失和保存模型，# 每 10 触发一次（save_steps=10）
    #             if self.step %self.config.save_steps == 0:
    #                 # ① 打印训练 loss
    #                 tqdm.write(f'[Epoch {epoch+1}] Step: {self.step}] Train_Loss: {this_loss:.4f}')
    #                 self.writer.add_scalar('train/loss', this_loss, self.step)
    #                 # ② 跑验证集，拿到验证 loss
    #                 metrics=self.evaluate()
    #                 tqdm.write(f'Evaluation—Loss: {metrics}')
    #                 # ③ 判断是否早停 + 保存最佳模型
    #                 self.writer.add_scalar('val/loss', metrics['loss'], self.step)
    #                 # ④ 保存检查点（断点续训用）
    #                 if self._should_early_stop(metrics['loss']):
    #                     tqdm.write('早停！')
    #                     return
    #                 self._save_checkpoint()
    #             self.step+=1
    #     self.writer.close()
    #核心方法：验证评估，返回评价指标字典
    def evaluate(self):
        self.model.eval()
        total_loss=0
        total_tokens=0

        with torch.no_grad():
            for batch_data in  tqdm(self.val_loader,desc='评估'):
                batch=Batch(
                    batch_data['src'].to(self.device),
                    batch_data['tgt'].to(self.device),
                    pad_id=self.config.pad_id,
                )
                # 2、前向传播,传入中文、中文填充、英文、英文填充和因果填充
                out = self.model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
                out = self.model.generator(out)

                # 3、计算损失
                loss = self.loss_fn(
                    out.reshape(-1, out.size(-1)),
                    batch.tgt_y.reshape(-1),
                )
                total_loss += loss.item()
                total_tokens += batch.ntokens.item()
        return {'loss':total_loss/max(1,total_tokens)}

if __name__ == '__main__':
    train()
