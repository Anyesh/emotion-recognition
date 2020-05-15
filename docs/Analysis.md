# Problem Formulation

|                      | Prompt                     | Answer                                                                                                                                  |
| -------------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Task**             | We want a model that can   | analyse the user emotion and classify their sentiment.                                                                                  |
| **Experience**       | Using                      | the text feedback which are provided by the users.                                                                                      |
| **Performance**      | As Measured by             | testing on the validation data and checking the accuracy/f1 score, self checking the output of the trained model by using custom input. |
| **Reason to solve**  | The output will be used to | know what customer thinks about the product or company, how we can improve the user experience.                                         |
| **Success Criteria** | It's a success if          | the model has an accuracy/f1 score over 50% on test data.                                                                               |

---

# Solution Formulation

| Prompt                                    | Answer                                                                                                                                             |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Manually, the problem could be solved as: | Looking at each feedbacks manually and analyzing its sentiment.                                                                                    |
| It can be formulated as a ML Problem as:  | NLP, Classification problem.                                                                                                                       |
| A similar ML task is:                     | Natural language processing, Movie review classification, Twitter sentiment classification.                                                        |
| Our assumptions are                       | Users will always give their openion on the product as a feedback, Data are labelled.                                                              |
| A baseline approach could be:             | To start with, we will create a tf-idf vectorized dataset from the corpus and train a Naive Bayes Classifier on that for sentiment classification. |
