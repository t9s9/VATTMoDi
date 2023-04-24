def make_batches(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
