import os
import pandas as pd
import json
import numpy as np
import joblib
import spacy
import time
import matplotlib.pyplot as plt
import re

from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.sparse import csc_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer

def old():
    return None

    def tfidf(df, path, sources, chunkQueries, results, norm, n, k):
        if not os.path.exists(path):
            os.mkdir(path)

        # ----- Prepare tfidf-matrix ------

        data_orig = df['corpus'].tolist()
        model = TfidfVectorizer(norm=norm)
        model.fit(data_orig)
        X = model.transform(data_orig)

        # ----- Lemmatization -----

        TOKENIZER = True
        if TOKENIZER:
            if "en" in path:
                spacyModel = "en_core_web_sm"
            elif "de" in path:
                spacyModel = "de_core_news_sm"

            nlp = spacy.load(spacyModel, disable=["parser", "ner"])
            df['lemma'] = df['corpus'].apply(lambda row: " ".join([w.lemma_ for w in nlp(row)]))
            data = df['lemma'].tolist()

        # ----- Create Index -----

        SPECF = 0.9
        vectorizer = TfidfVectorizer(norm=norm, max_df = SPECF)
        X = vectorizer.fit_transform(data)
        df = pd.DataFrame(X.T.toarray(), index=vectorizer.get_feature_names_out())

        # ----  Save Index -----

        if norm == 'l2':
            fn_mod = os.path.join(path, "vectorizer_L2.pkl")
            fn_tfidf = os.path.join(path, "table_L2.pkl")
            fn_keywords = os.path.join(path, f"keywords_{k}_L2.json")
            fn_queryResults = os.path.join(path, f"query_result_{n}_TFIDF_Opt_L2.json")
        
        elif norm == 'l1':
            fn_mod = os.path.join(path, "vectorizer_L1.pkl")
            fn_tfidf = os.path.join(path, "table_L1.pkl")
            fn_keywords = os.path.join(path, f"keywords_{k}_L1.json")
            fn_queryResults = os.path.join(path, f"query_result_{n}_TFIDF_Opt_L1.json")

        else:    
            fn_mod = os.path.join(path, "vectorizer.pkl")         
            fn_tfidf = os.path.join(path, "table.pkl")            
            fn_keywords = os.path.join(path, f"keywords_{k}.json")   
            fn_queryResults = os.path.join(path, f"query_result_{n}_TFIDF_Opt.json")

        joblib.dump(vectorizer, fn_mod)
        joblib.dump(df, fn_tfidf) 

        # ----- Load Index ----

        vectorizer = joblib.load(fn_mod)
        tfidf = joblib.load(fn_tfidf)

        # ----- Keyword Extraction -----

        KEYWORD_EXTRACTION = False
        if KEYWORD_EXTRACTION:
            start = time.time()
            unstacked = tfidf.unstack()

            largest_values = unstacked.nlargest(k)
            end = time.time()
            print(end - start)

            with open(fn_keywords, "w", encoding="utf-8") as f:
                json.dump([{"token": row, "document": sources[col], "score": 1 - value} for (col, row), value in largest_values.items()], f, indent=4, ensure_ascii=False)
        
        # ----- Query Data ---- 

        QUERY_RESULT = True
        if QUERY_RESULT:
            queryResults(chunkQueries, data_orig, vectorizer, df, n, nlp, sources, fn_queryResults, path, results)

    def init_tfidf(norm, n, k):

        with open("chunksSource.json", "r", encoding="utf-8") as f:
            chunkListpy = json.load(f)

        with open("queries.json", "r", encoding="utf-8") as f:
            chunkQueries = json.load(f)

        df_en = pd.DataFrame({'corpus': [chunk["text"] for chunk in chunkListpy if chunk["lang"] == "en"]})
        df_en.index.name='id'

        df_de = pd.DataFrame({'corpus': [chunk["text"] for chunk in chunkListpy if chunk["lang"] == "de"]})
        df_de.index.name='id' 

        results = {"en": [], "de": []}
        tfidf(df_en, "data/en", [chunk["source"] for chunk in chunkListpy if chunk["lang"] == "en"], chunkQueries, results, norm, n, k)
        tfidf(df_de, "data/de", [chunk["source"] for chunk in chunkListpy if chunk["lang"] == "de"], chunkQueries, results, norm, n, k)

    def queryResultIteration(mylist, sources, data_orig, n):

        best_list = sorted(mylist, key=lambda tup: tup[1], reverse=True)[:n]
        result = []

        for count, x in enumerate(best_list):
            if x[1] > 0.0:
                score = 1 - x[1]
                #score_calibrated_l2_exact_en = 0.1306 * score + 0.0758
                #score_calibrated_l2_general_en = 0.0232 * score + 0.1506
                #score_calibrated_l2_linked = 0.0769 * score + 0.1132
                #score_calibrated_noNorm_exact_en = 0.0001 * score + 0.1833
                #score_calibrated_noNorm_general_en = 0.000009 * score + 0.1686 

                #score_calibrated_l2_exact_de = 0.0664 * score + 0.0983
                #score_calibrated_l2_general_de = 0.0510 * score + 0.1185
                #score_calibrated_noNorm_exact_de = 0.0001 * score + 0.1721
                #score_calibrated_noNorm_general_de = 0.000045 * score + 0.1649
        
                document = sources[x[0]]
                content = data_orig[x[0]]
                print(f'Treffer {count+1} (score von {score}) mit Dokumentlink: {document}\n{content}\n')
                result.append({"document": document, "score": score, "content": content})
            else:
                print('Keine weiteren Treffer gefunden')
                break
        
        return result

    def queryResults(chunkQueries, data_orig, vectorizer, df, n, nlp, sources, fn_queryResults, path, results):

        for chunkq in chunkQueries:
            if chunkq["lang"] == path[5:]:
                data_q = [" ".join([w.lemma_ for w in nlp(chunkq["text"])])]

                q_vec = vectorizer.transform(data_q)
                search_results = (q_vec @ df)[0]
                si = search_results.argsort().tolist()[::-1]
                mylist = list(zip(si, search_results[si]))

                print(f'\nLemmatisierte Anfrage: {data_q}\n')

                result = queryResultIteration(mylist, sources, data_orig, n) 
                results[chunkq["lang"]].append({"query": chunkq["text"], "documents": result})

        if path[5:] == "en":
            with open(fn_queryResults, "w", encoding="utf-8") as f:
                json.dump(results["en"], f, indent=4, ensure_ascii=False)

        if path[5:] == "de":
            with open(fn_queryResults, "w", encoding="utf-8") as f:
                json.dump(results["de"], f, indent=4, ensure_ascii=False)

    def plotValues(x_values, y_values, width, height):

        plt.figure(figsize=(width, height))
        plt.scatter(x_values, y_values, color='blue', label='Score points')

        plt.xlabel('TF-IDF Scores')
        plt.ylabel('RAG Scores')
        plt.title('Pairwise Scores Scatter Plot')
        plt.legend()

        plt.grid(True)
        plt.show()

    def queryExactMatch(query1, query2, n):
        
        x_values = []
        y_values = []

        for i in range(len(query1)):
            for j in range(0, n):
                documents1 = query1[i].get("documents", [])
                doc1 = documents1[j].get("document")

                documents2 = query2[i].get("documents", [])
                doc2 = documents2[j].get("link")

                if doc1 == doc2:
                    query_ = query1[i].get("query")
                    score1 = documents1[j].get("score")
                    x_values.append(score1)
                    score2 = documents2[j].get("score")
                    y_values.append(score2)
                    print(f"Übereinstimmung gefunden bei: {query_}\nDokument: {doc1}\nPlatzierung: {j+1}\nTF-IDF Score: {score1}\nRAG Score: {score2}")
                    print()

        print(x_values)
        print(y_values)
        plotValues(x_values, y_values, 10, 6)
        MSElinRegression(x_values, y_values)

    def queryGeneralMatch(query1, query2, n):
        
        x_values = []
        y_values = []

        for i in range(len(query1)):
            for j in range(0, n):
                documents1 = query1[i].get("documents", [])
                doc1 = documents1[j].get("document")

                documents2 = query2[i].get("documents", [])
                
                for k in range(0, n):
                    if doc1 == documents2[k].get("link"):
                        query_ = query1[i].get("query")
                        score1 = documents1[j].get("score")
                        x_values.append(score1)
                        score2 = documents2[k].get("score")
                        y_values.append(score2)
                        print(f"Übereinstimmung gefunden bei: {query_}\nDokument: {doc1}\nPlatzierung: TFIDF: {j+1} und RAG: {k+1}\nTF-IDF Score: {score1}\nRAG Score: {score2}")
                        print()

        plotValues(x_values, y_values, 10, 6)
        MSElinRegression(x_values, y_values)

    def linInterpolate(x_values, y_values):
        
        sorted_indices = np.argsort(x_values)
        x_sorted = np.array(x_values)[sorted_indices]
        y_sorted = np.array(y_values)[sorted_indices]

        linear_interp = interp1d(x_sorted, y_sorted, kind='linear')

        x_new = np.linspace(min(x_sorted), max(x_sorted), 300)
        y_new = linear_interp(x_new)

        plt.figure(figsize=(10, 6))
        plt.scatter(x_values, y_values, color='red', label='Original Data')
        plt.plot(x_new, y_new, label='Linear Interpolation', color='blue')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Linear Interpolation')
        plt.legend()
        plt.grid(True)
        plt.show()

    def MSElinRegression(x_values, y_values):

        x = np.array(x_values).reshape(-1, 1)
        y = np.array(y_values)

        model = LinearRegression()
        model.fit(x, y)
        y_pred = model.predict(x)

        mse = mean_squared_error(y, y_pred)
        print(f"Mean Squared Error: {mse}")

        m = model.coef_[0]
        b = model.intercept_
        equation = f"y = {m:.6f}x + {b:.4f}"
        print(f"Mathematical Equation of best-fit line: {equation}")

        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color='red', label='Original Data')
        plt.plot(x, y_pred, label='Linear Regression', color='blue')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Linear Regression (MSE: {mse:.5f})')
        plt.legend()
        plt.grid(True)
        plt.show()

class Tfidf:
    
    def __init__(self, chunkListpy):   

        self.chunkListpy = chunkListpy

    def buildTfidf(self, norm, path):

        self.norm = norm
        self.path = path

        if not os.path.exists(self.path):
            os.mkdir(self.path)

        if 'en' in self.path:
            df = pd.DataFrame({'corpus': [chunk["chunks"] for chunk in self.chunkListpy if chunk["lang"] == "en"]})
            df.index.name='id'
            spacyModel = "en_core_web_sm"
            self.sources = [chunk["link"] for chunk in self.chunkListpy if chunk["lang"] == "en"]

        elif 'de' in self.path:
            df = pd.DataFrame({'corpus': [chunk["chunks"] for chunk in self.chunkListpy if chunk["lang"] == "de"]})
            df.index.name='id'
            spacyModel = "de_core_news_sm"
            self.sources = [chunk["link"] for chunk in self.chunkListpy if chunk["lang"] == "de"]


        # ----- Prepare tfidf-matrix ------

        self.data_orig = df['corpus'].tolist()
        model = TfidfVectorizer(norm=self.norm)
        model.fit(self.data_orig)
        X = model.transform(self.data_orig)


        # ----- Lemmatization -----

        self.nlp = spacy.load(spacyModel, disable=["parser", "ner"])
        df['lemma'] = df['corpus'].apply(lambda row: " ".join([w.lemma_ for w in self.nlp(row)]))
        data = df['lemma'].tolist()

        SPECF = 0.9
        vectorizer = TfidfVectorizer(norm=self.norm, max_df = SPECF)
        X = vectorizer.fit_transform(data)
        df = pd.DataFrame(X.T.toarray(), index=vectorizer.get_feature_names_out())

        if self.norm == None:
            self.norm = 'N'

        # ------ Save Index ------

        if 'l' in self.norm:
            fn_mod = os.path.join(self.path, f"vectorizer_{self.norm}.pkl")
            fn_tfidf = os.path.join(self.path, f"table_{self.norm}.pkl")

        else:    
            fn_mod = os.path.join(self.path, "vectorizer.pkl")         
            fn_tfidf = os.path.join(self.path, "table.pkl")            

        joblib.dump(vectorizer, fn_mod)
        joblib.dump(df, fn_tfidf)        

        # ----- Load Index -----

        self.vectorizer = joblib.load(fn_mod)
        self.tfidf = joblib.load(fn_tfidf)

    def keywordExtract(self, k):
        #----- Keyword Extraction -----#
        self.k = k

        if 'l' in self.norm:
            self.fn_keywords = os.path.join(self.path, f"keywords_{k}_{self.norm}.json")
        else:
            self.fn_keywords = os.path.join(self.path, f"keywords_noNorm_{self.k}.json") 
        
        start = time.time()
        unstacked = self.tfidf.unstack()

        largest_values = unstacked.nlargest(self.k)
        end = time.time()
        print(end - start)

        with open(self.fn_keywords, "w", encoding="utf-8") as f:
            json.dump([{"token": row, "document": self.sources[col], "score": 1 - value} for (col, row), value in largest_values.items()], f, indent=4, ensure_ascii=False)

    def queryResult(self, chunkQueries, n):
        #----- Query Result -----#
        self.n = n

        if 'l' in self.norm:
            self.fn_queryResults = os.path.join(self.path, f"class_query_result_{self.n}_TFIDF_Opt_{self.norm}.json")
        else:
            self.fn_queryResults = os.path.join(self.path, f"query_result_{self.n}_TFIDF_Opt_noNorm.json")

        results = {"en": [], "de": []}
        for chunkq in chunkQueries:
            if chunkq["lang"] == self.path[5:]:
                data_q = [" ".join([w.lemma_ for w in self.nlp(chunkq["text"])])]

                q_vec = self.vectorizer.transform(data_q)
                search_results = (q_vec @ self.tfidf)[0]
                si = search_results.argsort().tolist()[::-1]
                self.mylist = list(zip(si, search_results[si]))

                print(f'\nLemmatisierte Anfrage: {data_q}\n')

                result = self.queryResultIteration()
                results[chunkq["lang"]].append({"query": chunkq["text"], "documents": result})

        if 'en' in self.path:
            with open(self.fn_queryResults, "w", encoding="utf-8") as f:
                json.dump(results["en"], f, indent=4, ensure_ascii=False)

        if 'de' in self.path:
            with open(self.fn_queryResults, "w", encoding="utf-8") as f:
                json.dump(results["de"], f, indent=4, ensure_ascii=False)

    def queryResultIteration(self):
        #----- Query Result Iteration -----#
        best_list = sorted(self.mylist, key=lambda tup: tup[1], reverse=True)[:self.n]
        result = []

        for count, x in enumerate(best_list):
            if x[1] > 0.0:
                score = 1 - x[1]

                if 'en' in self.path:
                    if 'l2' in self.norm:
                        fscore = 0.1306 * score + 0.0758 # score_calibrated_l2_exact_en
                        #score_calibrated_l2_general_en = 0.0232 * score + 0.1506
                        #score_calibrated_l2_linked = 0.0769 * score + 0.1132
                    else:    
                        fscore = 0.0001 * score + 0.1833 # score_calibrated_noNorm_exact_en
                        #score_calibrated_noNorm_general_en = 0.000009 * score + 0.1686 

                elif 'de' in self.path:
                    if 'l2' in self.norm:
                        fscore = 0.0664 * score + 0.0983 # score_calibrated_l2_exact_de
                        #score_calibrated_l2_general_de = 0.0510 * score + 0.1185
                    else:
                        fscore = 0.0001 * score + 0.1721 # score_calibrated_noNorm_exact_de
                        #score_calibrated_noNorm_general_de = 0.000045 * score + 0.1649
        
                document = self.sources[x[0]]
                content = self.data_orig[x[0]]
                print(f'Treffer {count+1} (score von {fscore}) mit Dokumentlink: {document}\n{content}\n')
                result.append({"document": document, "score": fscore, "content": content})
            else:
                print('Keine weiteren Treffer gefunden')
                break
        
        return result

    def queryExactMatches(self, query2):
        #----- Exact Matches -----#
        self.mode = 'exact'

        with open(self.fn_queryResults, "r", encoding="utf-8") as f:
            query1 = json.load(f)
        
        self.x_values = []
        self.y_values = []
        self.score1_values = []
        self.score2_values = []

        for i in range(len(query1)):
            for j in range(0, self.n):
                documents1 = query1[i].get("documents", [])
                doc1 = documents1[j].get("document")

                documents2 = query2[i].get("documents", [])
                doc2 = documents2[j].get("link")

                if doc1 == doc2:
                    query_ = query1[i].get("query")
                    score1 = documents1[j].get("score")
                    self.x_values.append(score1)
                    score2 = documents2[j].get("score")
                    self.y_values.append(score2)

                    if score1 > score2:
                        self.score1_values.append(score1)
                    else:
                        self.score2_values.append(score2)

                    print(f"Übereinstimmung gefunden bei: {query_}\nDokument: {doc1}\nPlatzierung: {j+1}\nTF-IDF Score: {score1}\nRAG Score: {score2}")
                    print()

        print(self.x_values)
        print(self.y_values)
        self.MSElinRegression()
        self.barGraphValueDiff()

    def queryGeneralMatches(self, query2):
        #----- General Matches -----#
        self.mode = 'general'

        with open(self.fn_queryResults, "r", encoding="utf-8") as f:
            query1 = json.load(f)

        self.x2_values = []
        self.y2_values = []

        for i in range(len(query1)):
            for j in range(0, self.n):
                documents1 = query1[i].get("documents", [])
                doc1 = documents1[j].get("document")

                documents2 = query2[i].get("documents", [])
                
                for k in range(0, self.n):
                    if doc1 == documents2[k].get("link"):
                        query_ = query1[i].get("query")
                        score1 = documents1[j].get("score")
                        self.x2_values.append(score1)
                        score2 = documents2[k].get("score")
                        self.y2_values.append(score2)
                        print(f"Übereinstimmung gefunden bei: {query_}\nDokument: {doc1}\nPlatzierung: TFIDF: {j+1} und RAG: {k+1}\nTF-IDF Score: {score1}\nRAG Score: {score2}")
                        print()

        print(self.x2_values)
        print(self.y2_values)
        self.MSElinRegression()

    def MSElinRegression(self):
        #----- Linear Regression Graph -----#
        x = np.array(self.x_values).reshape(-1, 1)
        y = np.array(self.y_values)

        model = LinearRegression()
        model.fit(x, y)
        y_pred = model.predict(x)

        mse = mean_squared_error(y, y_pred)
        print(f"Mean Squared Error: {mse}")

        m = model.coef_[0]
        b = model.intercept_
        equation = f"y = {m:.6f}x + {b:.4f}"
        print(f"Mathematical Equation of best-fit line: {equation}")

        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color='red', label='Original Data')
        plt.plot(x, y_pred, label='Linear Regression', color='blue')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Linear Regression (MSE: {mse:.5f})')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"data\{self.path[5:]}\graphs\{self.mode}\Pairwise_Scatter_Plot_TFIDF_RAG_Scores_{self.path[5:]}_{self.mode}_{self.n}.jpg")
        plt.show()

    def barGraphValueDiff(self):
        #----- Bar Graph -----#
        fig, ax = plt.subplots()

        mechanism = ['TFIDF', 'RAG']
        counts = [len(self.score1_values), len(self.score2_values)]
        bar_labels = ['red', 'blue']
        bar_colors = ['tab:red', 'tab:blue']

        ax.bar(mechanism, counts, label=bar_labels, color=bar_colors)

        ax.set_ylabel('Number of higher values')
        ax.set_title('Precision after calibration')

        plt.show()

    def query(self, query_, lang='en', maxChunks=4, threshold=0.25):

        self.n = maxChunks
        #----- Lemmatization & Matrix Multiplication -----#
        if 'en' in lang:
            data_q = [" ".join([w.lemma_ for w in self.nlp_en(query_)])]
            query_words = re.findall(r'\w+', data_q[0])

            q_vec = self.vectorizer_en.transform(data_q)
            feature_names = self.vectorizer_en.get_feature_names_out()
            search_results = (q_vec @ self.tfidf_en)[0]

            idf = self.vectorizer_en.idf_
            idf_df = pd.DataFrame(idf, index=feature_names, columns=["IDF"])

        elif 'de' in lang:

            data_q = [" ".join([w.lemma_ for w in self.nlp_de(query_)])]
            query_words = re.findall(r'\w+', data_q[0])

            feature_names = self.vectorizer_de.get_feature_names_out()
            q_vec = self.vectorizer_de.transform(data_q)
            search_results = (q_vec @ self.tfidf_de)[0]

            idf = self.vectorizer_de.idf_
            idf_df = pd.DataFrame(idf, index=feature_names, columns=["IDF"])

        else:
            return 'Language not supported.'

        si = search_results.argsort().tolist()[::-1]
        self.mylist = list(zip(si, search_results[si]))
        best_list = sorted(self.mylist, key=lambda tup: tup[1], reverse=True)[:self.n]
        result = []


        for count, x in enumerate(best_list):
            if x[1] > 0.0:

                #----- Score Calibration -----#
                self.score = 1 - x[1]
                if 'en' in lang:
                    #fscore = 0.1306 * score + 0.0758 # score_calibrated_l2_exact_en
                    #score_calibrated_l2_general_en = 0.0232 * score + 0.1506
                    #score_calibrated_l2_linked = 0.0769 * score + 0.1132

                    self.fscore = 0.0001 * self.score + 0.1833 # score_calibrated_noNorm_exact_en
                    #fscore = 0.000009 * score + 0.1686 #score_calibrated_noNorm_general_en

                elif 'de' in lang:
                    #fscore = 0.0664 * score + 0.0983 # score_calibrated_l2_exact_de
                    #score_calibrated_l2_general_de = 0.0510 * score + 0.1185
                    self.fscore = 0.0001 * self.score + 0.1721 # score_calibrated_noNorm_exact_de 
                    #score_calibrated_noNorm_general_de = 0.000045 * score + 0.1649

                #----- Word Analysis English-----#
                if self.fscore < threshold:
                    if 'en' in lang:
                        link = self.link_en[x[0]]
                        text = self.data_orig_en[x[0]]
                        documentID = self.documentID_en[x[0]]
                        origin = self.origin_en[x[0]]
                        referenceText = self.referenceText_en[x[0]]

                        wordAnalysis = []
                        for i in range(len(query_words)):
                            q_word = query_words[i]
                            q_word_vec = self.vectorizer_en.transform(q_word.split())
                            search_results_word = (q_word_vec @ self.tfidf_en)[0]
                            si_word = search_results_word.argsort().tolist()[::-1]
                            self.mylist_word = list(zip(si_word, search_results_word[si_word]))
                            average_value = sum([tup[1] for tup in self.mylist_word]) / len(self.mylist_word)

                            if x[0] in search_results_word.nonzero()[0]:
                            
                                word_value_indoc = [y[1] for y in self.mylist_word if y[0] == x[0]][0]
                                word_indoc_index = [y[0] for y in self.mylist_word if y[0] == x[0]][0]
                                idf_value = idf_df.loc[f"{q_word.lower()}", "IDF"]           
                                wordAnalysis.append({f"{q_word}": {"value in document": word_value_indoc, "average value across all documents": average_value, "word count": word_value_indoc / (idf_value * idf_value), "idf value": idf_value, "document index valid?": word_indoc_index, "document index": x[0]}})

                            elif len(search_results_word.nonzero()[0]) > 0:

                                wordAnalysis.append({f"{q_word}": "no occurence" })

                            else:

                                wordAnalysis.append({f"{q_word}": "stopword" })

                            idf_value = 0
                        

                    #----- Word Analysis German-----#
                    elif 'de' in lang:
                        link = self.link_de[x[0]]
                        text = self.data_orig_de[x[0]]
                        documentID = self.documentID_de[x[0]]
                        origin = self.origin_de[x[0]]
                        referenceText = self.referenceText_de[x[0]]

                        wordAnalysis = []
                        for i in range(len(query_words)):
                            q_word = query_words[i]
                            q_word_vec = self.vectorizer_de.transform(q_word.split())
                            search_results_word = (q_word_vec @ self.tfidf_de)[0]
 
                            si_word = search_results_word.argsort().tolist()[::-1]
                            self.mylist_word = list(zip(si_word, search_results_word[si_word]))
                            average_value = sum([tup[1] for tup in self.mylist_word]) / len(self.mylist_word)

                            if x[0] in search_results_word.nonzero()[0]:
                            
                                word_value_indoc = [y[1] for y in self.mylist_word if y[0] == x[0]][0]
                                word_indoc_index = [y[0] for y in self.mylist_word if y[0] == x[0]][0]
                                idf_value = idf_df.loc[f"{q_word.lower()}", "IDF"]           
                                wordAnalysis.append({f"{q_word}": {"value in document": word_value_indoc, "average value across all documents": average_value, "word count": word_value_indoc / (idf_value * idf_value), "idf value": idf_value, "document index valid?": word_indoc_index, "document index": x[0]}})

                            elif len(search_results_word.nonzero()[0]) > 0:

                                wordAnalysis.append({f"{q_word}": "no occurence" })

                            else:

                                wordAnalysis.append({f"{q_word}": "stopword" })
                            idf_value = 0

                    result.append({"documentID": documentID, "lang": lang, "link": link, "origin": origin, "referenceText": referenceText, "score": self.fscore, "wordAnalysis": wordAnalysis, "text": text})   

            else:
                print('Keine weiteren Treffer gefunden')
                break

        print(json.dumps(result, indent=4))  
             
    def setup(self):

        # ----- Build Corpus & Prep  ----- #
        self.chunks_en = []
        self.link_en = []
        self.origin_en = []
        self.referenceText_en = []
        self.documentID_en = []
        
        for chunk in self.chunkListpy:
            if chunk["lang"] == "en":
                for i in range(len(chunk["chunks"])):
                    self.chunks_en.append(chunk["chunks"][i])   
                    self.link_en.append(chunk["link"])
                    self.origin_en.append(chunk["origin"])
                    self.referenceText_en.append(chunk["referenceText"])
                    self.documentID_en.append(chunk["documentID"])       
    
        df_en = pd.DataFrame({'corpus': self.chunks_en})
        df_en.index.name='id'

        self.chunks_de = []
        self.link_de = []
        self.origin_de = []
        self.referenceText_de = []
        self.documentID_de = []
        for chunk in self.chunkListpy:
            if chunk["lang"] == "de":
                for i in range(len(chunk["chunks"])):
                    self.chunks_de.append(chunk["chunks"][i])
                    self.link_de.append(chunk["link"])
                    self.origin_de.append(chunk["origin"])
                    self.referenceText_de.append(chunk["referenceText"])
                    self.documentID_de.append(chunk["documentID"])

        df_de = pd.DataFrame({'corpus': self.chunks_de})
        df_de.index.name='id'

        stop_words_en = ['why', 'what', 'how', 'when', 'who'] 
        self.data_orig_en = df_en['corpus'].tolist()

        stop_words_de = ['was', 'wie', 'warum', 'wieso', 'weshalb', 'wer']
        self.data_orig_de = df_de['corpus'].tolist()

        self.vectorizer_de = TfidfVectorizer(max_df=0.9, norm=None, stop_words=stop_words_de)
        self.vectorizer_en = TfidfVectorizer(max_df=0.9, norm=None, stop_words=stop_words_en)

        # ----- Lemmatization ----- #       
        spacyModel_de = "de_core_news_sm"
        self.nlp_de = spacy.load(spacyModel_de, disable=["parser", "ner"])
        df_de['lemma'] = df_de['corpus'].apply(lambda row: " ".join([w.lemma_ for w in self.nlp_de(row)]))
        data_de = df_de['lemma'].tolist()
        self.X_de = self.vectorizer_de.fit_transform(data_de)
        df_de = pd.DataFrame(self.X_de.T.toarray(), index=self.vectorizer_de.get_feature_names_out())

        spacyModel_en = "en_core_web_sm"
        self.nlp_en = spacy.load(spacyModel_en, disable=["parser", "ner"])
        df_en['lemma'] = df_en['corpus'].apply(lambda row: " ".join([w.lemma_ for w in self.nlp_en(row)]))
        data_en = df_en['lemma'].tolist()
        self.X_en = self.vectorizer_en.fit_transform(data_en)
        df_en = pd.DataFrame(self.X_en.T.toarray(), index=self.vectorizer_en.get_feature_names_out())         

        self.tfidf_en = df_en
        self.tfidf_de = df_de

        # ----- Stoppwörter ----- #

        # model_en_sw = TfidfVectorizer(norm=None, min_df=0.9)
        # model_en_sw.fit_transform(data_en)
        # self.stopwords_en = model_en_sw.get_feature_names_out()
        # print(self.stopwords_en)

        # model_de_sw = TfidfVectorizer(norm=None, min_df=0.9)
        # model_de_sw.fit_transform(data_de)
        # self.stopwords_de = model_de_sw.get_feature_names_out()
        # print(self.stopwords_de)

    def saveToCSV(self):
        self.tfidf_en.to_csv("data/en/table_en.csv", sep=';', decimal=',')
        self.tfidf_de.to_csv("data/de/table_de.csv", sep=';', decimal=',')

if __name__ == '__main__':
    
    # with open("data/en/query_result_20_TFIDF_Opt_L2.json", "r", encoding="utf-8") as f:
    #     query_results_tfidf_en = json.load(f)

    # with open("data/de/query_result_20_TFIDF_Opt.json", "r", encoding="utf-8") as f:
    #     query_results_tfidf_de = json.load(f)

    # with open("data/en/query_result_RAG_20.json", "r", encoding="utf-8") as f:
    #     query_results_rag_en = json.load(f)

    # with open("data/de/query_result_RAG_20.json", "r", encoding="utf-8") as f:
    #     query_results_rag_de = json.load(f)

    with open("request.json", "r", encoding="utf-8") as f:
        chunkListpy = json.load(f)

    # with open("queries.json", "r", encoding="utf-8") as f:
    #     chunkQueries = json.load(f)

    tfidf_test = Tfidf(chunkListpy)

    tfidf_test.setup()

    #tfidf_test.saveToCSV()

    print("Finished setup...")
    
    tfidf_test.query("What is PYTHIA?", 'en', 7, 0.20)

    #tfidf_test.buildTfidf(None, "data\de")
    #tfidf_test.keywordExtract(10) # 390s for k = 10
    #tfidf_test.queryResult(chunkQueries, 20)
    #tfidf_test.queryExactMatches(query_results_rag_de)


