import numpy as np

def eval_dataset(inference_model, dataset, threshold):
    good = 0
    false_positive = 0
    false_negative = 0
    for x in range(len(dataset)):
        for y in range(x, (len(dataset))):
            x_image, x_writer = dataset[x]
            y_image, y_writer = dataset[y]

            decision = inference_model.is_match(x_image.unsqueeze(0), y_image.unsqueeze(0), threshold=threshold)
            if (x_writer == y_writer and decision == True) or (x_writer != y_writer and decision == False):
                good += 1
            else:
                if decision == True:
                    false_positive += 1
                else:
                    false_negative += 1

    return good, false_positive, false_negative

def get_best_threshold(inference_model, dataset):
    thresholds = np.linspace(0,1,10)
    res = []
    for threshold in thresholds:
        res.append(eval_dataset(inference_model, dataset,threshold))

    n = sum([sum(x) for x in res])
    good = [x[0] for x in res]
    
    i = good.index(max(good))

    return thresholds[i]