import numpy as np
from munch import DefaultMunch

rng = np.random.default_rng(9)  # manually seed random number generator

def interval_generator(dist):
    intervals = []

    if dist.mode == "even_space": # evenly spaced gaps
        for k in range(dist.dims):
            intervals.append([[(2 * i) / (2 * dist.intervals[k] - 1), (2 * i + 1) / (2 * dist.intervals[k] - 1)] for i in range(0, dist.intervals[k])])
            intervals[k] = np.array(intervals[k])
            intervals[k] = intervals[k] * (dist.max[k] - dist.min[k]) + dist.min[k]
    elif dist.mode == "explicit": # explicitly described gaps
        for k in range(dist.dims):
            intervals.append(np.array(dist.intervals[k]))
    else:
        raise ValueError("dist interval specification unrecognized")

    return intervals

def in_intervals(dist, data):
    intervals = interval_generator(dist)

    data = np.transpose(data)

    is_interval = []
    for k in range(dist.dims):
        is_interval.append(np.logical_and(data[k][:, np.newaxis] > intervals[k][np.newaxis, :, 0], data[k][:, np.newaxis] < intervals[k][np.newaxis, :, 1]))
        is_interval[k] = np.any(is_interval[k], axis=1)
    is_interval = np.array(is_interval)

    if dist.dims == 1 or dist.combine == "all":
        is_interval = np.all(is_interval, axis=0)
    elif dist.combine == "any":
        is_interval = np.any(is_interval, axis=0)
    else:
        raise ValueError("dist.combine unrecognized")
    return is_interval

def sample_distribution(dist, num):
    if dist.type == "uniform":
        ret = []

        if dist.dims == 1:
            dist.max = [dist.max]
            dist.min = [dist.min]

        for dim in range(dist.dims):
            ret.append(rng.uniform(dist.min[dim], dist.max[dim], size=num))
        
        if dist.dims == 1:
            dist.max = dist.max[0] # Undo modifications
            dist.min = dist.min[0]
            return ret[0]
        else:
            return np.stack(ret, axis=-1)
    elif dist.type == "uniform_with_intervals":
        # Note: this is uniform across the union of all intervals,
        # i.e. if some intervals are longer than other intervals,
        # they will be more likely.

        intervals = interval_generator(dist)

        final = np.zeros((0, dist.dims))
        num_increased = 10000
        while np.shape(final)[0] < num:
            num_to_generate = num - np.shape(final)[0]
            if num_increased < 0.1 * num_to_generate:
                num_to_generate *= 100
            elif num_increased < 0.5 * num_to_generate:
                num_to_generate *= 10

            ret = []
            for k in range(dist.dims):
                ret.append(rng.uniform(np.min(intervals[k]), np.max(intervals[k]), size=num_to_generate))

            is_interval = in_intervals(dist, np.transpose(np.array(ret)))

            ret = np.swapaxes(np.array(ret), 0, 1)
            ret = ret[is_interval]
            num_increased = ret.shape[0]

            final = np.concatenate((final, ret))
            print(f"Interval generation: Generated {final.shape[0]} out of {num}...")
        final = final[:num]

        return final
    elif dist.type == "exponential":
        if dist.dims == 1:
            return dist.shift + rng.exponential(dist.scale, size=num)
        else:
            raise NotImplementedError
    elif dist.type == "stack":
        ret = []
        for smalld in dist.dists:
            ret.append(sample_distribution(smalld, num))
            if len(ret[-1].shape) == 1:
                ret[-1] = ret[-1][:, np.newaxis]

        ret = np.concatenate(ret, axis=1)

        if dist.reorder_axes != None:
            if len(dist.reorder_axes) != ret.shape[1]:
                raise ValueError("reorder axes not of correct length")
            ret = ret[:, dist.reorder_axes]

        return ret
    elif dist.type == "single":
        return np.array([dist.value] * num)
    elif dist.type == "combine":
        arrs = []

        if sum(dist.ratio) < 0.95 or sum(dist.ratio) > 1:
            raise ValueError("ratios must sum to 1")

        for i, smalld in enumerate(dist.dists):
            arrs.append(sample_distribution(smalld, int(num * dist.ratio[i] * 1.1)))
        ret = np.concatenate(arrs, axis=0)
        rng.shuffle(ret, axis=0)
        ret = ret[:num]

        return ret
    elif dist.type == "discrete":
        if sum(dist.ratio) != 1:
            raise ValueError("ratios must sum to 1")

        bars = np.cumsum([0] + dist.ratio[:-1])
        random = rng.uniform(0, 1, size=num)
        idx = np.searchsorted(bars, random)

        return np.array(dist.values)[idx]
    else:
        raise NotImplementedError # implement other kinds of distributions

def is_in_distribution(config_or_dist, arr):
    reduce_to_one = False
    if isinstance(arr, int) or isinstance(arr, float):
        reduce_to_one = True
        arr = [arr]

    if len(arr.shape) == 1:
        arr = arr[:, np.newaxis]

    if not isinstance(config_or_dist, str) and "type" in vars(config_or_dist):
        dist = config_or_dist
    else:
        if isinstance(config_or_dist, str):
            config = read_config(config_or_dist)

        for key, value in vars(config).items():
            if isinstance(value, dict) and "traj_distr" in value:
                dist = DefaultMunch.fromDict(value["traj_distr"], object())

    if dist.type == "uniform":
        omin, omax = dist.min, dist.max
        if isinstance(dist.min, float) or isinstance(dist.min, int):
            dist.min = [dist.min]
            dist.max = [dist.max]

        dist.min = np.array(dist.min)
        dist.max = np.array(dist.max)
        ret = np.logical_and(arr > dist.min[np.newaxis, :], arr < dist.max[np.newaxis, :])
        ret = np.all(ret, axis=1)
        dist.min, dist.max = omin, omax
    elif dist.type == "uniform_with_intervals":
        ret = in_intervals(dist, arr)
    elif dist.type == "exponential":
        oshift, oscale = dist.shift, dist.scale
        if isinstance(dist.shift, float) or isinstance(dist.shift, int):
            dist.shift = [dist.shift]
            dist.scale = [dist.scale]

        dist.shift = np.array(dist.shift)
        dist.scale = np.array(dist.scale)
        ret = np.all(np.logical_and(arr > dist.shift[np.newaxis, :], arr < (dist.shift[np.newaxis, :] + dist.scale[np.newaxis, :])), axis=1)
        dist.shift, dist.scale = oshift, oscale
    elif dist.type == "stack":
        if dist.reorder_axes != None:
            inverted = [dist.reorder_axes.index(i) for i in range(len(dist.reorder_axes))]
            arr = arr[:, inverted]

        startid = 0
        in_dists = []
        for smalld in dist.dists:
            if smalld.type == "single":
                if isinstance(smalld.value, int) or isinstance(smalld.value, float):
                    smalldims = 1
                else:
                    smalldims = len(smalld.value)
            else:
                smalldims = smalld.dims
            in_dists.append(is_in_distribution(smalld, arr[:, startid:startid + smalldims]))
            startid += smalldims

        ret = np.all(np.array(in_dists), axis=0)
        #ret = np.array(in_dists)
    elif dist.type == "single":
        if isinstance(dist.value, int) or isinstance(dist.value, float):
            value = np.array([dist.value])
        else:
            value = dist.value

        ret = np.equal(arr, np.array(value)[np.newaxis, :])
        ret = np.all(ret, axis=1)
    elif dist.type == "combine":
        in_dists = []
        for i, smalld in enumerate(dist.dists):
            if dist.ratio[i] > 0.0:
                in_dists.append(is_in_distribution(smalld, arr))

        ret = np.any(in_dists, axis=0)
        return ret
    else:
        raise NotImplementedError

    if reduce_to_one:
        ret = ret[0]

    return ret

def indistribution_noise(vals, dist, noise, repeat=1):
    """
        Adds noise to values while keeping them in-distribution.
    """
    
    ret = []
    for _ in range(repeat):
        nse = np.zeros(vals.shape)
        done = np.full(vals.shape, False)
        while not np.all(done):
            cand = noise * rng.standard_normal(size=vals.shape)
            mask = is_in_distribution(dist, vals + cand)
            if len(vals.shape) > 1:
                mask = np.repeat(mask[:, np.newaxis], vals.shape[1], axis=1)
            cand = np.where(mask, cand, np.zeros(vals.shape))

            nse = np.where(done, nse, nse + cand)
            done = np.logical_or(done, mask)

        ret.append(nse)

    return np.stack(ret, axis=-1)
