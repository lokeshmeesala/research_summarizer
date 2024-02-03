import arxiv

client = arxiv.Client()


def get_articles(topic):
    search = arxiv.Search(
    #   query = "attention mechanisms",
    query = topic,
    max_results = 5
    )
    results = client.results(search)
    abstracts = [(i+1, r.summary.replace("\n"," "),r.entry_id) for i,r in enumerate(results)]
    return abstracts



if __name__ == "__main__":
    test_topic1 = "attention mechanisms"
    test_topic2 = "LLM Context Length Extension",
    abstracts = get_articles(topic=test_topic1)
    print(abstracts)