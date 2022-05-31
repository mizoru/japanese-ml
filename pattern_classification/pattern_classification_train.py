import os
from fastai.vision.all import *
from fastaudio.core.all import *
from fastaudio.augment.all import *
from fastcore.xtras import untar_dir
from fastcore.script import *
from fastai.basics import *
from fastai.callback.all import *
from fastprogress import fastprogress
from torchvision.models import *
from fastai.vision.models.xresnet import *
from fastai.callback.mixup import *
from fastcore.script import *
import wandb
from fastai.callback.wandb import WandbCallback

def download_data(dest = Path('data')):
    if not dest.exists():
        dataset = Path('dataset')
        if not dataset.exists():
            os.system("git clone https://gitlab.com/mizoru/dataset.git")
        tarfile.open(Path('dataset/pitch_accent.tar.gz'), "r:gz").extractall(data)
        dataset.unlink()
    corrupt_file = dest/('pitch_accent/dict2/見上げさせる-1567_8_2_male.mp3')
    if corrupt_file.exists():
        corrupt_file.unlink()
    return dest / 'pitch_accent'

def prepare_labels(p):
    labels_p = p/'labels.csv'
    if labels_p.exists():
        labels = pd.read_csv(labels_p)
    else:
        labels1 = pd.read_csv(p/'dict1_labels.csv')
        labels1 = labels1[labels1.pattern.isin(['頭高', '中高', '平板'])]
        labels1.path = 'dict1/' + labels1.path
        labels2 = pd.read_csv(p/'dict2_labels.csv')
        labels2 = labels2[labels2.pattern.isin(['頭高', '中高', '平板'])]
        labels2 = labels2[labels2.path != '見上げさせる-1567_8_2_male.mp3']
        labels2.path = 'dict2/' + labels2.path
        labels = pd.concat([labels1, labels2])
        labels.to_csv(labels_p, index=False)
    labels.path = str(p) + '/' + labels.path
    labels['is_valid'] = False
    labels.loc[labels.type == 'dict2 male', 'is_valid'] = True
    return labels

def get_x(df):
    return df.path
def get_y(df):
    return df.pattern

def alter_learner(learn):
    layer = learn.model[0][0]
    layer.in_channels = 1
    layer.weight = nn.Parameter(layer.weight[:,1,:,:].unsqueeze(1))
    learn.model[0][0] = layer
    return learn


def get_dls(df, item_tfms, fp16=True, shuffle=True, bs=32, num_workers=0):
    new_dblock = DataBlock(blocks=[AudioBlock, CategoryBlock],
                  item_tfms=item_tfms,
                #   batch_tfms=SpectrogramNormalize,
                  get_x=get_x,
                  get_y=get_y,
                  splitter=ColSplitter())
    new_dls = new_dblock.dataloaders(df, shuffle=shuffle, bs=bs, num_workers=num_workers)
    return new_dls

def get_learner(dls, model_func, wd, fp16, act_fn, sa, sym, pool, pretrained, cbs):
    model = model_func(pretrained=pretrained, n_out=3, act_cls=act_fn, sa=sa, sym=sym, pool=pool)
    new_learn = Learner(dls, model, CrossEntropyLossFlat(), 
                metrics=[accuracy, F1Score(average='weighted')], wd=wd, cbs=cbs)
    if fp16:
        new_learn = new_learn.to_fp16()
    new_learn = alter_learner(new_learn)
    new_learn.unfreeze()
    return new_learn

@call_parse
def main(
    lr:    Param("Learning rate", float)=1e-2,
    sqrmom:Param("sqr_mom", float)=0.99,
    mom:   Param("Momentum", float)=0.9,
    eps:   Param("Epsilon", float)=1e-6,
    wd:    Param("Weight decay", float)=1e-2,
    epochs:Param("Number of epochs", int)=5,
    bs:    Param("Batch size", int)=64,
    mixup: Param("Mixup", float)=0.,
    opt:   Param("Optimizer (adam,rms,sgd,ranger)", str)='ranger',
    arch:  Param("Architecture", str)='xresnet34',
    pretrained: Param("Use pretrained model", store_true)=True,
    sa:    Param("Self-attention", store_true)=False,
    sym:   Param("Symmetry for self-attention", int)=0,
    beta:  Param("SAdam softplus beta", float)=0.,
    act_fn:Param("Activation function", str)='Mish',
    fp16:  Param("Use mixed precision training", store_true)=True,
    pool:  Param("Pooling method", str)='AvgPool',
    n_mels:Param("Number of melbank filter", int)=128,
    dump:  Param("Print model; don't train", int)=0,
    runs:  Param("Number of times to repeat training", int)=1,
    meta:  Param("Metadata (ignored)", str)='',
    workers:   Param("Number of workers", int)=None
):
    config = {k:v for k,v in locals().copy().items() if k[:2] != '__'}
    print(config)
    if   opt=='adam'  : opt_func = partial(Adam, mom=mom, sqr_mom=sqrmom, eps=eps)
    elif opt=='rms'   : opt_func = partial(RMSprop, sqr_mom=sqrmom)
    elif opt=='sgd'   : opt_func = partial(SGD, mom=mom)
    elif opt=='ranger': opt_func = partial(ranger, mom=mom, sqr_mom=sqrmom, eps=eps, beta=beta)

    print(f'epochs: {epochs}; lr: {lr}; sqrmom: {sqrmom}; mom: {mom}; eps: {eps}')
    model_func,act_fn,pool = [globals()[o] for o in (arch,act_fn,pool)]

    path = download_data()
    labels = prepare_labels(path)
    item_tfms = [RemoveSilence(), ResizeSignal(2000, AudioPadType.Zeros), AudioToSpec.from_cfg(AudioConfig.Voice(f_min=0, n_mels=n_mels, normalized=True))]
    dls = get_dls(labels, item_tfms, bs=bs, num_workers=workers)
    cbs = MixUp(mixup) if mixup else []
    name = f'{arch}_bs-{bs}_lr-{lr}_{opt}_mels-{n_mels}_mom-{mom}_{("","pretrained")[pretrained]}'
    for run in range(runs):
        run_name = name + f'_{run}'
        wandb.init(project='pattern_classification', name=run_name+f'_{run}', config=config)
        print(f'Run: {run}')
        learn = get_learner(dls, model_func, wd, fp16, act_fn, sa, sym, pool, pretrained, cbs=cbs+[WandbCallback()])
        if dump: print(learn.model); exit()
        learn.fit_flat_cos(epochs, lr, wd=wd, cbs=cbs)
        learn.save(run_name, with_opt=True)
        wandb.save('/models'+run_name+'.pth', policy='now')
        COMMON_WORDS = ["もう","わかる","バック","社会","読む","入る","来る","トラック","によって","仕事","同じ","数","記事","いただく","彼","大","国","等","くださる","回","三","とか","君","法","K","意味","力","以上","J","会社","j","よる","ほど","そんな","人間","現在","作る","企業","氏","ちょっと","間","可能","感じる","出す","研究","投稿","他","アメリカ","しれる","けれども","リンク","今回","いたす","高い","次","ら","言葉","こういう","おく", "わたし","熱く","深い"]
        prediction_labels = labels[labels.type=='dict2 male']
        prediction_labels[prediction_labels["path"].str.contains("|".join('/' + word + '[\.-]' for word in COMMON_WORDS))]
        test_dl = learn.dls.test_dl(prediction_labels[:2])
        print(learn.get_preds(dl=test_dl, with_decoded=True, with_input=True))
        wandb.finish()


if __name__ == "__main__":
    main()