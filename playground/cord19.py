from ir_datasets import load

d = load("cord19/fulltext")
text_len = 0
abstract_len = 0
est_size = min(9000000, d.docs_count())
for i, doc in enumerate(d.docs_iter()):
    abstract_len += len(doc.abstract.split(" "))
    text_len += sum([len(section.text.split(" ")) for section in doc.body])
    if i % 10000 == 0:
        print(f"Iteration: {i}")
    if i >= est_size:
        break


print(f"Abstract len {abstract_len/est_size}")
print(f"Text len {text_len/est_size}")
