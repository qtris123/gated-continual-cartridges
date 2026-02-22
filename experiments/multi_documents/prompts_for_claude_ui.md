# Prompts for Claude UI

Upload both 10-K documents (AMD_2022_10K.txt and PEPSICO_2022_10K.txt), then paste one prompt at a time. Save each response as `data/responses/{name}.txt` (the filename in brackets).

Each prompt asks for 16 Q&A pairs. The expected output format is:
```
1. Q: <question>
   A: <answer>
```

After saving all responses, run:
```bash
python experiments/multi_documents/parse_responses_to_pkl.py
```

---

## 1. Factual  [Save as: `factual.txt`]

```
I have uploaded two 10-K annual report documents (AMD 2022 and PepsiCo 2022). Using BOTH documents as context, please generate 16 unique questions to test someone's ability to remember factual details from BOTH documents. The answer should be a few tokens long and be a factual detail from the document, such as a number, entity, date, title, or name. These questions should not be common knowledge: instead, they should be something that is only answerable via information in the documents. Include questions that draw from both documents.

Here are a few examples of the kind of cross-document questions I expect:

1. Q: Who served as the independent auditor for Company A and Company B, respectively?
   A: [Auditor X] audited Company A and [Auditor Y] audited Company B.

2. Q: What were the total revenues reported by Company A and Company B for fiscal year 2022?
   A: Company A reported approximately $X billion and Company B reported approximately $Y billion.

3. Q: In which states are Company A and Company B incorporated?
   A: Company A is incorporated in [State X] and Company B is incorporated in [State Y].

Now generate 16 new questions in this style. Please format your response exactly as:

1. Q: <question>
   A: <answer>

2. Q: <question>
   A: <answer>

...and so on for all 16 pairs.
```

---

## 2. Knowledge  [Save as: `knowledge.txt`]

```
I have uploaded two 10-K annual report documents (AMD 2022 and PepsiCo 2022). Using BOTH documents as context, please generate 16 unique questions that require combining information mentioned both inside and outside the documents. Each question should require using a fact from the documents and also a fact that you are confident about, but is not mentioned in the documents. For instance: "What are the founding dates of the companies that got acquired this year?" is a good question because the names of the acquired companies are mentioned in the document and the founding dates are not mentioned. "What is the name of the CEO's spouse?" is a good question because the name of the CEO is mentioned in the document and the spouse's name is not mentioned. The answer should be a fact that is a few tokens long such as a number, entity, date, title, or name. Include questions that draw from both documents.

Here are a few examples of the kind of cross-document questions I expect:

1. Q: In what cities are the headquarters of Company A and Company B located, and what are those cities' approximate populations?
   A: Company A is headquartered in [City X] (~N people) and Company B in [City Y] (~M people).

2. Q: Both Company A and Company B trade on a major stock exchange. In what year was that exchange founded, and what are each company's ticker symbols?
   A: [Exchange] was founded in [year]. Company A trades as "[TICKER1]" and Company B trades as "[TICKER2]."

Now generate 16 new questions in this style. Please format your response exactly as:

1. Q: <question>
   A: <answer>

2. Q: <question>
   A: <answer>

...and so on for all 16 pairs.
```

---

## 3. Disjoint  [Save as: `disjoint.txt`]

```
I have uploaded two 10-K annual report documents (AMD 2022 and PepsiCo 2022). Using BOTH documents as context, please generate 16 unique multi-hop questions that test someone's ability to use factual information mentioned in at least two very different sub-sections of the documents, or across both documents. These questions shouldn't be standard questions about this kind of document. Instead, they should ask about two particularly disconnected ideas, like comparing information about the amount of owned space for one company's headquarters with the amount of dollars of estimated liability for the other, or comparing a revenue number from one company with the number of employees at the other. These questions should also test one's ability to do retrieval: do not give away part of the answer in the question. Ensure that for one to get the correct answer to the question, they need to understand the documents. The answer should be short: for example, a number, entity, date, title, or name.

Here are a few examples of the kind of cross-document questions I expect:

1. Q: Compare Company A's total goodwill with Company B's total long-term debt — which is larger and by how much?
   A: Company B's long-term debt exceeds Company A's goodwill by roughly $X billion.

2. Q: If you added Company A's total R&D expenses to Company B's advertising and marketing costs, what would the combined amount be?
   A: Approximately $X billion combined.

Now generate 16 new questions in this style. Please format your response exactly as:

1. Q: <question>
   A: <answer>

2. Q: <question>
   A: <answer>

...and so on for all 16 pairs.
```

---

## 4. Synthesize  [Save as: `synthesize.txt`]

```
I have uploaded two 10-K annual report documents (AMD 2022 and PepsiCo 2022). Using BOTH documents as context, please generate 16 unique questions that require synthesizing and aggregating information in the documents. For instance, you could ask someone to summarize a page of one of the documents, list all the key competitors mentioned across both documents, compare the two companies' business models, or aggregate financial data from both filings. Include questions that draw from both documents.

Here are a few examples of the kind of cross-document questions I expect:

1. Q: Compare the key risk factors highlighted by Company A and Company B in their respective 10-K filings. What themes are shared and what differs?
   A: Both cite [shared risks]. Company A additionally emphasizes [risk X], while Company B highlights [risk Y].

2. Q: List a few competitors for each of Company A and Company B as stated in each 10-K.
   A: Company A lists [competitor 1] and [competitor 2]. Company B lists [competitor 3], [competitor 4], and others.

Now generate 16 new questions in this style. Please format your response exactly as:

1. Q: <question>
   A: <answer>

2. Q: <question>
   A: <answer>

...and so on for all 16 pairs.
```

---

## 5. Structure  [Save as: `structure.txt`]

```
I have uploaded two 10-K annual report documents (AMD 2022 and PepsiCo 2022). Using BOTH documents as context, please generate 16 unique questions that require understanding the structure of the documents. These questions should be more about the structure of the documents, rather than the precise content details. For instance, you could ask someone to list the titles of all the sections in one document, describe how the document structures differ between the two filings, report the total number of pages, ask which section amongst two sections comes first, report the section with the largest number of tables, or compare how both companies organize their risk factors. Include questions that draw from both documents.

Here are a few examples of the kind of cross-document questions I expect:

1. Q: How do Company A and Company B differ in how they organize their business segment reporting sections?
   A: Company A organizes segments by [approach X], while Company B organizes segments by [approach Y].

2. Q: Do both Company A and Company B include a "Risk Factors" section in Part I, and roughly how do the lengths compare?
   A: Yes, both include Risk Factors under Part I, Item 1A. Company B's section is [longer/shorter] due to [reason].

Now generate 16 new questions in this style. Please format your response exactly as:

1. Q: <question>
   A: <answer>

2. Q: <question>
   A: <answer>

...and so on for all 16 pairs.
```

---

## 6. Creative  [Save as: `creative.txt`]

```
I have uploaded two 10-K annual report documents (AMD 2022 and PepsiCo 2022). Using BOTH documents as context, please generate 16 unique questions about the documents to test someone's ability to comprehend the content of the documents. These questions specifically should be focused on their ability to generalize the information about the documents to strange or unusual questions. These questions shouldn't be standard questions about this kind of document—they should ask to do something abnormal and creative, like writing a poem about a financial document, comparing the two companies in an unexpected way, or creating an analogy between the companies' strategies and something unrelated. Include questions that draw from both documents.

Here are a few examples of the kind of cross-document questions I expect:

1. Q: Write a short dialogue between the CEOs of Company A and Company B comparing their biggest challenges in 2022.
   A: CEO A: "[challenge from filing]." CEO B: "[challenge from filing]." CEO A: "[shared concern]."

2. Q: Imagine a joint venture between Company A and Company B. Based on each company's strengths in their 10-Ks, what product or service might they create together?
   A: They could develop [product idea], combining Company A's [strength] with Company B's [strength].

Now generate 16 new questions in this style. Please format your response exactly as:

1. Q: <question>
   A: <answer>

2. Q: <question>
   A: <answer>

...and so on for all 16 pairs.
```

---

## 7. Counting  [Save as: `counting.txt`]

```
I have uploaded two 10-K annual report documents (AMD 2022 and PepsiCo 2022). Using BOTH documents as context, please generate 16 unique questions that require counting how frequently different events occur in the documents. These questions should be about statistical properties of the documents, rather than the content details. For instance, you could ask someone to count the number of times the word "million" is mentioned in one document, count the length of the shortest section title, compare how many times certain terms appear across both documents, or count the number of risk factors listed. The answer should be a number. Include questions that draw from both documents.

Here are a few examples of the kind of cross-document questions I expect:

1. Q: Which document mentions the word "revenue" more frequently — Company A's or Company B's 10-K?
   A: Company B's 10-K mentions "revenue" more frequently, with approximately N occurrences versus M for Company A.

2. Q: How many reportable business segments does Company A disclose compared to Company B?
   A: Company A discloses X segments while Company B discloses Y segments.

Now generate 16 new questions in this style. Please format your response exactly as:

1. Q: <question>
   A: <answer>

2. Q: <question>
   A: <answer>

...and so on for all 16 pairs.
```

---

## 8. Reasoning  [Save as: `reasoning.txt`]

```
I have uploaded two 10-K annual report documents (AMD 2022 and PepsiCo 2022). Using BOTH documents as context, please generate 16 unique questions that require mathematical reasoning over the values in the documents. These questions should require going beyond the facts directly mentioned in the documents, such as asking to compute the percentage increase in revenue between two years, find the largest expense category, calculate the difference in profit between the two companies, compare growth rates, or derive ratios from the financial data. The answer should be a number. Include questions that draw from both documents.

Here are a few examples of the kind of cross-document questions I expect:

1. Q: What is the difference in net revenue between Company A and Company B for fiscal year 2022?
   A: Company B's net revenue exceeds Company A's by approximately $X billion.

2. Q: Based on each company's reported capital expenditures and total revenue, which company invests a higher percentage of revenue in capex?
   A: Company B invests approximately X% of revenue in capex versus Company A's Y%, making Company B the higher spender as a percentage of revenue.

Now generate 16 new questions in this style. Please format your response exactly as:

1. Q: <question>
   A: <answer>

2. Q: <question>
   A: <answer>

...and so on for all 16 pairs.
```
