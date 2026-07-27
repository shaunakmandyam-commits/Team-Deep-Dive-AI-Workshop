def transfer_weights(policy, old_algo, new_algo):
        
    old_weights = old_algo.get_weights()[policy]
    new_weights = new_algo.get_weights()[policy]
    weights = {}
    for (name, new_weights), (name2, old_weights) in zip(new_weights.items(), old_weights.items()):
        print(name, new_weights.shape)
        print(name2, old_weights.shape)
        new_weights.fill(0)

        if new_weights.shape != old_weights.shape:
            new_weights[:, :old_weights.shape[1]] = old_weights
        else:
            new_weights = old_weights
            
        weights[name] = new_weights
        print(new_weights[0])
    return weights
