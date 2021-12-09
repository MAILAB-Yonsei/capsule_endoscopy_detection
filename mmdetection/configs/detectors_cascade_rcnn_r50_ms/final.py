# final.py
_base_ = [
    'detectors_cascade_rcnn_r50_1x_coco.py',
    'dataset.py',
    'schedule_1x.py',
    'default_runtime.py'
]

runner = dict(type='EpochBasedRunner', max_epochs=14)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)

checkpoint_config = dict(interval=1)

optimizer = dict(type='SGD', lr=0.02 / 8, momentum=0.9, weight_decay=0.0001)
find_unused_parameters = True