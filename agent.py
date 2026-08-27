from search import search


def agent(query):
    results = search(query)

    if not results:
        return {
            "message": "No matching catalogue records found."
        }

    return {
        "query": query,
        "results": results
    }


if __name__ == "__main__":
    while True:
        query = input("Enter query: ").strip()

        if not query:
            break

        print(agent(query))