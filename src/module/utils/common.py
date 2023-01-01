
def chunks(l, size=1000, verbose=False):
    """
    Returns a (start, end, total, list) tuple for each batch in the given iterable.

    Usage:
        articles = []
        for start, end, total, separated_articles in chunks(articles, size=1000):
            for article in separated_articles:
                print(article.body)
    """
    total = len(l)

    for start in range(0, total, size):
        end = min(start + size, total)
        if verbose:
            print(f"Now processing {start + 1} - {end} of {total}")
        yield start, end, total, l[start:end]
