import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probability_distribution = {}
    all_pages = set(corpus.keys())
    links = corpus[page]

    if not links:
        # If the page has no outgoing links, treat it as linking to all pages (including itself)
        links = all_pages

    for p in all_pages:
        probability_distribution[p] = (1 - damping_factor) / len(all_pages)
        if p in links:
            probability_distribution[p] += damping_factor / len(links)

    return probability_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_rank = {page: 0 for page in corpus}
    pages = list(corpus.keys())

    # Start by choosing a random page
    current_page = random.choice(pages)

    for _ in range(n):
        page_rank[current_page] += 1
        probabilities = transition_model(corpus, current_page, damping_factor)
        current_page = random.choices(
            population=list(probabilities.keys()),
            weights=list(probabilities.values()),
            k=1
        )[0]

    # Normalize the ranks to sum to 1
    total_samples = sum(page_rank.values())
    for page in page_rank:
        page_rank[page] /= total_samples

    return page_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    page_rank = {page: 1 / N for page in corpus}
    all_pages = set(corpus.keys())

    # Adjust corpus to handle pages with no outgoing links
    adjusted_corpus = {
        page: (links if links else all_pages)
        for page, links in corpus.items()
    }

    converged = False
    while not converged:
        new_rank = {}
        for page in corpus:
            rank = (1 - damping_factor) / N
            rank += damping_factor * sum(
                page_rank[linking_page] / len(adjusted_corpus[linking_page])
                for linking_page in adjusted_corpus
                if page in adjusted_corpus[linking_page]
            )
            new_rank[page] = rank

        # Check for convergence
        converged = all(
            abs(new_rank[page] - page_rank[page]) < 0.001
            for page in page_rank
        )
        page_rank = new_rank

    # Normalize to ensure ranks sum to 1
    total_rank = sum(page_rank.values())
    for page in page_rank:
        page_rank[page] /= total_rank

    return page_rank

if __name__ == "__main__":
    main()
