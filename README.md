# README #

This project has three approaches to solve the Single Document Summarization problem. 


LSA_summary.py : This algorithm uses the Latent Semantic analysis approach to solve the problem.

KMedoid_summarize.py : This algorithm uses sentence similarity and K-medoid clustering approach to solve the problem.

Lexrank.py : This algorithm uses tf-idf values based on training corpus, sentence similarity based on the tf-idf vectors and then a graph based Lex Rank approach to solve the problem.

baseline_summarizer.py : This is a baseline approach which picks first k sentences of a document(This is based on research done in the field of document summarization which says the first few sentences are the most relevant sentences)

Clustering.py : This algorithm builds relevant clusters from a training corpus. The cluster which is most similar to the test document will be used to to extract the idf values for the test document.

idf_model.py : This algorithm builds a model file containing relevant idf values for words in the training corpus.

plot_bar.py : This code analyzes the performance of the summarizers discussed above against some of the best online summarizer tools by plotting a bar graph for documents and their Fscores from ROGUE evaluation system.
