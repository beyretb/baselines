from baselines.herSmallMovesMulti.experiment.trainFinal import launch

params = [('FetchReach-v1', 200),
          ('FetchPush-v1',200),
          ('FetchSlide-v1', 200),
          ('FetchPickAndPlace-v1', 200),
          ('HandReach-v0', 200),
          ('HandManipulateBlock-v0', 200),
          ('HandManipulateBlockRotateXYZ-v0', 200),
          ('HandManipulateBlockRotateZ-v0', 200),
          ('HandManipulateBlockRotateParallel-v0', 200),
          ('HandManipulateBlockFull-v0', 200),
          ('HandManipulateEggRotate-v0', 200),
          ('HandManipulateEggFull-v0', 200),
          ('HandManipulatePenRotate-v0', 200),
          ('HandManipulatePenFull-v0', 200)]

logdir = '/home/ben/resultsHerSmallMovesMulti/'
num_cpu = 19
seed = 0
replay_strategy = 'future'
policy_save_interval = 5
clip_return = 1
fb = True
save_policies=True

for e in params:
    try:
        env = e[0]
        n_epochs = e[1]
        launch(env, logdir, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return, fb)
    except:
        print('{} not OK'.format(e[0]))
        continue