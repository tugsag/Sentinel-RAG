from agent import create_rag_chain
from data import prepare_data
import fly_patch
import click


@click.command()
def main():
    prompt = click.prompt("Enter input")

    collection = prepare_data('rag_sample_qas_from_kis.csv')

    rag_graph = create_rag_chain(collection)
    res = rag_graph.invoke(prompt)

if __name__ == '__main__':
    main()