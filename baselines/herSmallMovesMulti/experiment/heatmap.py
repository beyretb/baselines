import sys
sys.path.append('/home/ben/small-moves-her/')
import click
import numpy as np
import pickle
import matplotlib.pyplot as plt


from baselines import logger
from baselines.common import set_global_seeds
import baselines.herSmallMovesMulti.experiment.config as config
from baselines.herSmallMovesMulti.rollout import RolloutWorker

@click.command()
@click.argument('policy_file', type=str)
@click.option('--seed', type=int, default=0)
@click.option('--n_test_rollouts', type=int, default=10)
@click.option('--render', type=int, default=1)
def main(policy_file, seed, n_test_rollouts, render):
    set_global_seeds(seed)

    # Load policy.
    with open(policy_file, 'rb') as f:
        policy_opt = pickle.load(f)
    env_name = policy_opt.info['env_name']

    # Prepare params.
    params = config.DEFAULT_PARAMS
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params['env_name'] = env_name
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    o = pickle.load(open('o.pkl','rb'))

    o[0,3]=1.1
    o[0,4]=0.45
    o[0,5]=0.44
    g = np.array([[1.5,1.05,0.42]])

    policy = policy_opt.target_G
    vals = [policy.Q_tf]

    mesh = np.meshgrid(np.linspace(1.1,1.5,20),np.linspace(0.45,1.05,20))
    N=50
    res = np.zeros((N,N))
    i=0
    j=0
    for x in np.linspace(1.1,1.5,N):
        j=0
        for y in np.linspace(0.45,1.05,N):
            sg = np.array([[x,y,0.42]])
            o[0, 3] = x
            o[0, 4] = y
            feed = {
                policy.o_tf: o.reshape(-1, policy_opt.dimo),
                policy.g_tf: g.reshape(-1, policy_opt.dimg),
                policy.d_tf: sg.reshape(-1, policy_opt.dimg) - g.reshape(-1, policy_opt.dimg)
            }
            val = policy_opt.sess.run([policy.Q_tf],feed_dict=feed)
            res[i,j] = val[0][0,0]
            j+=1
        i+=1
        print(i)

    plt.clf()
    im= plt.imshow(res.T, extent=[0.45,1.05,1.5,1.1])
    clb = plt.colorbar(im,fraction=0.1)
    clb.set_label('G-values')#, rotation=270)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    print('ok')

if __name__ == '__main__':
    main()
