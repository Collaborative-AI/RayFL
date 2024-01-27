from config import cfg


def process_control():
    cfg['model_name'] = cfg['control']['model_name']
    cfg['task_name'] = cfg['control']['task_name']
    cfg['batch_size'] = int(cfg['control']['batch_size'])
    cfg['step_period'] = int(cfg['control']['step_period'])
    cfg['num_steps'] = int(cfg['control']['num_steps'])
    cfg['eval_period'] = int(cfg['control']['eval_period'])
    cfg['run_mode'] = cfg['control']['run_mode']
    make_data_name()
    if cfg['task_name'] in ['s2s', 'sc', 'clm', 't2i']:
        cfg['collate_mode'] = 'transformer'
        cfg['bart-base'] = {'max_length': 128}
        cfg['roberta-base'] = {'max_length': 128}
        cfg['gpt2'] = {'max_length': 128}
        if 'llama' in cfg['model_name']:
            cfg[cfg['model_name']] = {'max_length': 128}
    elif cfg['task_name'] in ['ic']:
        cfg['collate_mode'] = 'dict'
        cfg['model'] = {}
        cfg['model']['model_name'] = cfg['model_name']
        data_shape = {'MNIST': [1, 28, 28], 'FashionMNIST': [1, 28, 28], 'SVHN': [3, 32, 32], 'CIFAR10': [3, 32, 32],
                      'CIFAR100': [3, 32, 32]}
        target_size = {'MNIST': 10, 'FashionMNIST': 10, 'SVHN': 10, 'CIFAR10': 10, 'CIFAR100': 100}
        cfg['model']['data_shape'] = data_shape[cfg['data_name']]
        cfg['model']['target_size'] = target_size[cfg['data_name']]
        cfg['model']['linear'] = {}
        cfg['model']['mlp'] = {'hidden_size': 128, 'scale_factor': 2, 'num_layers': 2, 'activation': 'relu'}
        cfg['model']['cnn'] = {'hidden_size': [64, 128, 256, 512]}
        cfg['model']['resnet9'] = {'hidden_size': [64, 128, 256, 512]}
        cfg['model']['resnet18'] = {'hidden_size': [64, 128, 256, 512]}
        cfg['model']['wresnet28x2'] = {'depth': 28, 'widen_factor': 2, 'drop_rate': 0.0}
        cfg['model']['wresnet28x8'] = {'depth': 28, 'widen_factor': 8, 'drop_rate': 0.0}
        cfg['model']['resmlp'] = {'hidden_size': 256, 'num_blocks': 3}
    else:
        raise ValueError('Not valid task name')

    tag = cfg['tag']
    cfg[tag] = {}
    if cfg['task_name'] in ['s2s', 'sc', 'clm']:
        cfg[tag]['optimizer_name'] = 'AdamW'
        if cfg['ft_name'] == 'full':
            cfg[tag]['lr'] = 5e-6
        else:
            cfg[tag]['lr'] = 3e-4
        cfg[tag]['momentum'] = 0.9
        cfg[tag]['betas'] = (0.9, 0.999)
        cfg[tag]['weight_decay'] = 5e-4
        cfg[tag]['nesterov'] = True
        cfg[tag]['num_epochs'] = 40
        cfg[tag]['batch_size'] = {'train': cfg['batch_size'], 'test': cfg['batch_size']}
        cfg[tag]['scheduler_name'] = 'LinearAnnealingLR'
        cfg[tag]['warmup_ratio'] = 0.05
    elif cfg['task_name'] in ['ic']:
        cfg[tag]['optimizer'] = {}
        cfg[tag]['optimizer']['optimizer_name'] = 'SGD'
        cfg[tag]['optimizer']['lr'] = 1e-1
        cfg[tag]['optimizer']['momentum'] = 0.9
        cfg[tag]['optimizer']['betas'] = (0.9, 0.999)
        cfg[tag]['optimizer']['weight_decay'] = 5e-4
        cfg[tag]['optimizer']['nesterov'] = True
        cfg[tag]['optimizer']['batch_size'] = {'train': cfg['batch_size'], 'test': cfg['batch_size']}
        cfg[tag]['optimizer']['step_period'] = cfg['step_period']
        cfg[tag]['optimizer']['num_steps'] = cfg['num_steps']
        cfg[tag]['optimizer']['scheduler_name'] = 'CosineAnnealingLR'
        cfg[tag]['optimizer']['warmup_ratio'] = 0
    elif cfg['task_name'] in ['t2i']:
        cfg['collate_mode'] = 'dreambooth'
        cfg[tag]['optimizer_name'] = 'AdamW'
        cfg[tag]['lr'] = 1e-4
        cfg[tag]['momentum'] = 0.9
        cfg[tag]['betas'] = (0.9, 0.999)
        cfg[tag]['weight_decay'] = 5e-4
        cfg[tag]['nesterov'] = True
        cfg[tag]['num_epochs'] = 40
        cfg[tag]['batch_size'] = {'train': cfg['batch_size'], 'test': cfg['batch_size']}
        # all settings are from the peft example: https://github.com/huggingface/peft
        cfg[tag]['prior_loss_weight'] = 1
        cfg[tag]['resolution'] = 512
        cfg[tag]['num_class_image'] = 200

        cfg[tag]['noise_scheduler_name'] = 'DDPM'
        cfg[tag]['beta_start'] = 0.00085
        cfg[tag]['beta_end'] = 0.012
        cfg[tag]['beta_schedule'] = 'scaled_linear'
        cfg[tag]['num_train_timesteps'] = 1000

        cfg[tag]['scheduler_name'] = 'ConstantLR'
        cfg[tag]['factor'] = 1

        cfg[tag]['num_inference_steps'] = 50
        cfg[tag]['guidance_scale'] = 7.5
    else:
        raise ValueError('Not valid task name')
    return


def make_data_name():
    data_name_list = cfg['control']['data_name'].split('-')
    if len(data_name_list) == 2:
        cfg['data_name'], cfg['subset_name'] = data_name_list
    else:
        cfg['data_name'] = data_name_list[0]
        cfg['subset_name'] = 'none'
    if cfg['task_name'] in ['s2s', 'sc', 'clm', 't2i']:
        data_name_dict = {
            # https://huggingface.co/datasets/financial_phrasebank
            'fpb': {'data_name': 'financial_phrasebank',
                    'subset_name_dict': {'sa': {'subset_name': 'sentences_allagree',
                                                'text_column': 'sentence',
                                                'label_column': 'text_label'}}},
            # https://huggingface.co/datasets/ptb_text_only
            'ptb': {'data_name': 'ptb_text_only',
                    'subset_name_dict': {'none': {'subset_name': None,
                                                  'text_column': 'sentence',
                                                  'label_column': None}}},
            # https://huggingface.co/datasets/wikisql
            'wikisql': {'data_name': 'wikisql',
                        'subset_name_dict': {'none': {'subset_name': None,
                                                      'text_column': ['question', 'table'],
                                                      'label_column': 'sql'}}},
            # https://huggingface.co/datasets/samsum
            # https://paperswithcode.com/dataset/samsum-corpus
            # https://arxiv.org/src/1911.12237v2/anc
            'samsum': {'data_name': 'samsum',
                       'subset_name_dict': {'none': {'subset_name': None,
                                                     'text_column': 'dialogue',
                                                     'label_column': 'summary'}}},
            # https://huggingface.co/datasets/e2e_nlg
            'e2enlg': {'data_name': 'e2e_nlg',
                       'subset_name_dict': {'none': {'subset_name': None,
                                                     'text_column': 'meaning_representation',
                                                     'label_column': 'human_reference'}}},
            # https://huggingface.co/datasets/web_nlg
            'webnlg': {'data_name': 'web_nlg',
                       'subset_name_dict': {'2017': {'subset_name': 'webnlg_challenge_2017',
                                                     'text_column': ['category', 'modified_triple_sets'],
                                                     'label_column': 'lex'}}},
            # https://huggingface.co/datasets/dart
            'dart': {'data_name': 'dart',
                     'subset_name_dict': {'none': {'subset_name': None,
                                                   'text_column': 'hardcode, complex structure',
                                                   'label_column': 'hardcode, complex structure'}}},
            # https://huggingface.co/datasets/glue
            'glue': {'data_name': 'glue',
                     'subset_name_dict': {'cola': {'subset_name': 'cola',
                                                   'text_column': ['sentence'],
                                                   'label_column': 'label'},
                                          'mnli': {'subset_name': 'mnli',
                                                   'text_column': ['premise', 'hypothesis'],
                                                   'label_column': 'label'},
                                          'mrpc': {'subset_name': 'mrpc',
                                                   'text_column': ['sentence1', 'sentence2'],
                                                   'label_column': 'label'},
                                          'qnli': {'subset_name': 'qnli',
                                                   'text_column': ['question', 'sentence'],
                                                   'label_column': 'label'},
                                          'qqp': {'subset_name': 'qqp',
                                                  'text_column': ['question1', 'question2'],
                                                  'label_column': 'label'},
                                          'rte': {'subset_name': 'rte',
                                                  'text_column': ['sentence1', 'sentence2'],
                                                  'label_column': 'label'},
                                          'sst2': {'subset_name': 'sst2',
                                                   'text_column': ['sentence'],
                                                   'label_column': 'label'},
                                          'stsb': {'subset_name': 'stsb',
                                                   'text_column': ['sentence1', 'sentence2'],
                                                   'label_column': 'label'},  # regression
                                          # datasize is small - not reported in LORA paper
                                          'wnli': {'subset_name': 'wnli',
                                                   'text_column': ['sentence1', 'sentence2'],
                                                   'label_column': 'label'}
                                          }
                     },
            # https://huggingface.co/datasets/databricks/databricks-dolly-15k
            'dolly': {'data_name': 'databricks/databricks-dolly-15k',
                      'subset_name_dict': {'15k': {'subset_name': '15k',
                                                   'text_column': ['instruction', 'context'],
                                                   'label_column': 'response'}
                                           }

                      },
            # Dataset: https://github.com/google/dreambooth
            # DreamBooth paper: https://arxiv.org/pdf/2208.12242.pdf
            'dreambooth': {'data_name': 'DreamBooth',
                           'subset_name_dict': {
                               'backpack': {'subset_name': 'backpack',
                                            'class': 'backpack',
                                            'category': 'object'},
                               'backpack_dog': {'subset_name': 'backpack_dog',
                                                'class': 'backpack',
                                                'category': 'object'},
                               'bear_plushie': {'subset_name': 'bear_plushie',
                                                'class': 'stuffed animal',
                                                'category': 'toy'},
                               'berry_bowl': {'subset_name': 'berry_bowl',
                                              'class': 'bowl',
                                              'category': 'object'},
                               'can': {'subset_name': 'can', 'class': 'can', 'category': 'object'},
                               'candle': {'subset_name': 'candle', 'class': 'candle', 'category': 'object'},
                               'cat': {'subset_name': 'cat', 'class': 'cat', 'category': 'live object'},
                               'cat2': {'subset_name': 'cat2', 'class': 'cat', 'category': 'live object'},
                               'clock': {'subset_name': 'clock', 'class': 'clock', 'category': 'object'},
                               'colorful_sneaker': {'subset_name': 'colorful_sneaker',
                                                    'class': 'sneaker',
                                                    'category': 'object'},
                               'dog': {'subset_name': 'dog', 'class': 'dog', 'category': 'live object'},
                               'dog2': {'subset_name': 'dog2', 'class': 'dog', 'category': 'live object'},
                               'dog3': {'subset_name': 'dog3', 'class': 'dog', 'category': 'live object'},
                               'dog5': {'subset_name': 'dog5', 'class': 'dog', 'category': 'live object'},
                               'dog6': {'subset_name': 'dog6', 'class': 'dog', 'category': 'live object'},
                               'dog7': {'subset_name': 'dog7', 'class': 'dog', 'category': 'live object'},
                               'dog8': {'subset_name': 'dog8', 'class': 'dog', 'category': 'live object'},
                               'duck_toy': {'subset_name': 'duck_toy', 'class': 'toy', 'category': 'toy'},
                               'fancy_boot': {'subset_name': 'fancy_boot',
                                              'class': 'boot',
                                              'category': 'object'},
                               'grey_sloth_plushie': {'subset_name': 'grey_sloth_plushie',
                                                      'class': 'stuffed animal',
                                                      'category': 'toy'},
                               'monster_toy': {'subset_name': 'monster_toy',
                                               'class': 'toy',
                                               'category': 'toy'},
                               'pink_sunglasses': {'subset_name': 'pink_sunglasses',
                                                   'class': 'glasses',
                                                   'category': 'accessory'},
                               'poop_emoji': {'subset_name': 'poop_emoji',
                                              'class': 'toy',
                                              'category': 'toy'},
                               'rc_car': {'subset_name': 'rc_car', 'class': 'toy', 'category': 'toy'},
                               'red_cartoon': {'subset_name': 'red_cartoon',
                                               'class': 'cartoon',
                                               'category': 'object'},
                               'robot_toy': {'subset_name': 'robot_toy', 'class': 'toy', 'category': 'toy'},
                               'shiny_sneaker': {'subset_name': 'shiny_sneaker',
                                                 'class': 'sneaker',
                                                 'category': 'object'},
                               'teapot': {'subset_name': 'teapot', 'class': 'teapot', 'category': 'object'},
                               'vase': {'subset_name': 'vase', 'class': 'vase', 'category': 'object'},
                               'wolf_plushie': {'subset_name': 'wolf_plushie',
                                                'class': 'stuffed animal',
                                                'category': 'toy'}}
                           }
        }
        if cfg['data_name'] == 'dreambooth':
            cfg['unique_id'] = 'sks'
            cfg['unique_class'] = data_name_dict[cfg['data_name']]['subset_name_dict'][cfg['subset_name']]['class']
        else:
            cfg['hf_data_name'] = data_name_dict[cfg['data_name']]['data_name']
            cfg['hf_subset_name'] = data_name_dict[cfg['data_name']]['subset_name_dict'][
                cfg['subset_name']]['subset_name']
            cfg['text_column'] = data_name_dict[cfg['data_name']]['subset_name_dict'][
                cfg['subset_name']]['text_column']
            cfg['label_column'] = data_name_dict[cfg['data_name']]['subset_name_dict'][
                cfg['subset_name']]['label_column']
    return
