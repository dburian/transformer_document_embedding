import ir_datasets

dataset = ir_datasets.load('msmarco-document/dev')
relevance_counts = [0, 0]
for qrel in dataset.qrels_iter():
    relevance_counts[qrel.relevance] += 1

print(relevance_counts)

# i = 0
# for query in dataset.queries_iter():
#     print(query)
#     i += 1
#     if i > 40:
#         break

AVG_COUNT = 10000
j = AVG_COUNT
length = 0
for doc in dataset.docs_iter():
    length += len(doc.body)
    j -= 1
    if j == 0:
        break

print(f'Average length: {length/AVG_COUNT}')
