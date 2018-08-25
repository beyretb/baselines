import numpy as np


def make_sample_her_transitions(replay_strategy, replay_k, reward_fun):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    def _sample_her_transitions(episode_batch, batch_size_in_transitions, n_subgoals):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions
        n_steps_per_subgoal = int(T/n_subgoals)

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        end_subgoal_episodes = np.ceil(t_samples/n_steps_per_subgoal)*n_steps_per_subgoal
        future_offset = np.random.uniform(size=batch_size) * (end_subgoal_episodes - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['sg'][her_indexes] = future_ag

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'sg']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions

    def _sample_her_goals_transitions(episode_batch, batch_size_in_transitions, sample_method, reward_type):
        '''
        sample_method: (int) describes which type of sampling we want for subgoals, choices are:
            1. sample traces as is, take (s_{t-1}, sg_{t-1}, sg_{t}) not considering if goals reached or not (no HER)
            2. replace all sg by ag and then sample as for 1 (ie: we only consider reached subgoals
            3. replace only sg_{t-1} with ag_{t}
        reward_type: (int) describes which type of reward we want to associate to golas traces
            1. simple sum of all rewards during the two subgoals episodes (s_{t-1} -> sg_{t-1} -> sg_{t})
            2. Add penalty for not reaching subgoals ??
            3. find more

        '''

        n_subgoals = episode_batch['sg'].shape[1]
        rollout_batch_size = episode_batch['sg'].shape[0]-1
        batch_size = batch_size_in_transitions

        if sample_method==1:
            episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
            t_samples = np.random.randint(n_subgoals, size=batch_size)
            transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                           for key in episode_batch.keys()}
        elif sample_method==2:
            episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
            t_samples = np.random.randint(n_subgoals, size=batch_size)
            transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                           for key in episode_batch.keys()}
            her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
            # In this case we replace the subgoal by the actual achieeved goal in hindsight
            transitions['sg'][her_indexes] = episode_batch['ag'][episode_idxs[her_indexes],(t_samples+1)[her_indexes]]



        # First, replace sg_t that haven't been reached by ag_{t+1} so that all traces have an achieved subgoal
        # sg_not_reached = np.where(episode_batch['sg_success'] == 0)
        # episode_batch['sg'][:,sg_not_reached, :] = episode_batch['ag'][:,(sg_not_reached[0], sg_not_reached[1]+1), :]

        # Select which episodes and time steps to use.
        # episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        # t_samples = np.random.randint(n_subgoals, size=batch_size)
        # transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
        #                for key in episode_batch.keys()}
        #
        # # We now apply HER to the end goal, replacing
        # her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        # future_offset = np.random.uniform(size=batch_size) * (n_subgoals - t_samples)
        # future_offset = future_offset.astype(int)
        # future_t = (t_samples + future_offset)[her_indexes]
        # future_ag = episode_batch['sg'][episode_idxs[her_indexes], future_t]
        # transitions['g'][her_indexes] = future_ag

        return transitions

    return _sample_her_transitions, _sample_her_goals_transitions


    # def _sample_her_transitions(episode_batch, batch_size_in_transitions):
    #     """episode_batch is {key: array(buffer_size x T x dim_key)}
    #     """
    #     T = episode_batch['u'].shape[1]
    #     rollout_batch_size = episode_batch['u'].shape[0]
    #     batch_size = batch_size_in_transitions
    #
    #     # Select which episodes and time steps to use.
    #     episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
    #     t_samples = np.random.randint(T, size=batch_size)
    #     transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
    #                    for key in episode_batch.keys()}
    #
    #     # Select future time indexes proportional with probability future_p. These
    #     # will be used for HER replay by substituting in future goals.
    #     her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
    #     future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
    #     future_offset = future_offset.astype(int)
    #     future_t = (t_samples + 1 + future_offset)[her_indexes]
    #
    #     # Replace goal with achieved goal but only for the previously-selected
    #     # HER transitions (as defined by her_indexes). For the other transitions,
    #     # keep the original goal.
    #     future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
    #     transitions['g'][her_indexes] = future_ag
    #
    #     # Reconstruct info dictionary for reward  computation.
    #     info = {}
    #     for key, value in transitions.items():
    #         if key.startswith('info_'):
    #             info[key.replace('info_', '')] = value
    #
    #     # Re-compute reward since we may have substituted the goal.
    #     reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
    #     reward_params['info'] = info
    #     transitions['r'] = reward_fun(**reward_params)
    #
    #     transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
    #                    for k in transitions.keys()}
    #
    #     assert(transitions['u'].shape[0] == batch_size_in_transitions)
    #
    #     return transitions
