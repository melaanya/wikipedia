#include <string>
#include <functional>
#include <map>
#include <vector>
#include <fstream>
#include <iostream>
#include <set>
#include <list>
#include <limits>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <valarray>
#include <algorithm>
#include <omp.h>
#include "macros.cpp"
//#include <sys/time.h>
//#include <boost/math/distributions/hypergeometric.hpp>


using namespace std;


const int NUM_OF_DOCS = 2365436;  //  //742tk 
const int NUM_OF_CATEGS = 500000; // 
const int NUM_OF_TEST_DOCS = 452168;  // на один больше, с учетом того, что нулевой не использукм
const int REAL_NUM_OF_CATEGS = 325056;
const int NUM_OF_TERMS = 2085167;  // если считать terms, которых вообще нет, удобная константа для индексации
const int REAL_NUM_OF_TERMS = 1617899;
const int NUM_OF_PAIRS = 179682414;    // в прошлой версии почему-то был на 1 меньше??
const int NUM_OF_PAIRS_TEST = 33557113;
const int NUM_OF_EXPERIMENTS = 1;
const double EPS = 0.00001;
class Category
{
public:
	int numofthecategory;
	int numofdocs;
	vector <int> docs;
	map <int, int> terms;
};

class Document
{
public:
	int numofthedoc;
	map <int, int> terms;
	valarray <int> fastTerms, fastCounts;
	set <int> categories;
	double getSimilarity(Document & d);  // min
	double getSimilarityKnn(Document & d);  // min
	int categoryfrom;
};

struct InfoMetrics
{
	double metrics;
	int cat, doc;
	bool belong;

	bool operator==(InfoMetrics const & rhs) const
	{
		return abs(metrics - rhs.metrics) < EPS;
	}
};

struct EntropiaPair
{
	int doc;
	int categ;
	double entropia;
};


bool compare_bigger(InfoMetrics lhs, InfoMetrics rhs)
{
	return lhs.metrics > rhs.metrics;
}

vector <Document> docs(NUM_OF_DOCS + 1);
vector <Document> testdocs(NUM_OF_TEST_DOCS);
vector <Category> categs(NUM_OF_CATEGS);
vector <InfoMetrics> metrics(NUM_OF_PAIRS);
vector <int> randDocs(NUM_OF_EXPERIMENTS);
map <int, int> frequency;
vector <int> terms(NUM_OF_TERMS);
map <int, list <int>> featDocs;
map <int, list <int>> featTestDocs;
vector <vector <int>> predictions(NUM_OF_TEST_DOCS);

void percent(double& lastProcent, double newProcent, clock_t t0, bool& flagAlreadyPrintTime)
{
	//dP - через сколько процентов выводим новое значение
	double dP = 0.1;
	clock_t t1 = clock();
	double toc = (double)(t1 - t0) / CLOCKS_PER_SEC;
	if (((newProcent>0.01) || (toc>60)) && (!flagAlreadyPrintTime))
	{
		flagAlreadyPrintTime = true;
		double rab2 = toc / newProcent;
		if (rab2<60) cout << "EstT = " << rab2 << " sec. ";
		else
			if (rab2<3600) cout << "EstT = " << rab2 / 60 << " min. ";
			else cout << "EstT = " << rab2 / 3600 << " hours. ";
			cout.flush();
	}
	if (newProcent - lastProcent >= dP)
	{
		lastProcent = newProcent;
		cout << floor(newProcent * 100) << "% ";
		cout.flush();
		if (1 - lastProcent < dP) cout << endl;
	}
}


double Document::getSimilarityKnn(Document & d)
{
	int i1 = 0, i2 = 0;
	int n1 = fastTerms.size();
	int n2 = d.fastTerms.size();
	int common = 0;
	if (n1 <= 0 || n2 <= 0)
	{
		//cout << "n1 <= 0 || n2 <= 0" << n1 << " " << n2 << " " << numofthedoc << " " << d.numofthedoc << endl;
		return 0;
		//exit(1);
	}
	while (true)
	{
		if (fastTerms[i1] < d.fastTerms[i2])
		{
			while (i1 != n1 && fastTerms[i1] < d.fastTerms[i2])
			{
				i1++;
			}
			if (i1 == n1) break;
		}
		else
		{
			while (i2 != n2 && fastTerms[i1] > d.fastTerms[i2])
			{
				i2++;
			}
			if (i2 == n2) break;
		}
		if (fastTerms[i1] == d.fastTerms[i2])
		{
			common++;
			++i1; ++i2;
		}
		if (i1 == n1 || i2 == n2) break;
	}
	return (n1 + n2 - 2.0 * common) / (n1 + n2 - common);
}


double Document::getSimilarity(Document & d)
{
	double res = 0;
	int i1 = 0, i2 = 0;
	int n1 = fastTerms.size();
	int n2 = d.fastTerms.size();
	if (n1 <= 0 || n2 <= 0)
	{
		//cout << "n1 <= 0 || n2 <= 0" << n1 << " " << n2 << " " << numofthedoc << " " << d.numofthedoc << endl;
		return 0;
		//exit(1);
	}
	while (true)
	{
		if (fastTerms[i1] < d.fastTerms[i2])
		{
			while (i1 != n1 && fastTerms[i1] < d.fastTerms[i2])
			{
				i1++;
			}
			if (i1 == n1) break;
		}
		else
		{
			while (i2 != n2 && fastTerms[i1] > d.fastTerms[i2])
			{
				i2++;
			}
			if (i2 == n2) break;
		}
		if (fastTerms[i1] == d.fastTerms[i2])
		{
			//double toadd = min(fastCounts[i1], d.fastCounts[i2]);
			res += 1;
			++i1; ++i2;
		}
		if (i1 == n1 || i2 == n2) break;
	}
	return res / this->terms.size();
}

// чтение и обработка исходного файла
void read_all_docs_without_catterms(ifstream & fin)
{
	int k = 0;
	fin.ignore(numeric_limits<streamsize>::max(), '\n');
	double lastProcent = 0;
	clock_t t0 = clock();
	bool flagAlreadyPrintTime = false;
	while (!fin.eof())
	{
		Document newdoc;
		newdoc.numofthedoc = k;
		char ch = ',';
		while (ch != ' ' && !fin.eof())
		{
			int a;
			fin >> a;
			newdoc.categories.insert(a);
			categs[a].numofthecategory = a;
			++categs[a].numofdocs;
			fin.get(ch);
			categs[a].docs.push_back(k); // номер документа
		}
		while (ch == ' ')
		{
			int feat, value;
			fin >> feat;
			fin.get(ch);
			fin >> value;
			newdoc.terms[feat] = value;
			featDocs[feat].push_back(k);
			fin.get(ch);
			/*if (feat >= NUM_OF_TERMS - 1)
				cout << feat << ": value = " << value << ", doc = " << k << endl;*/
		}
		newdoc.fastTerms.resize(newdoc.terms.size());
		newdoc.fastCounts.resize(newdoc.terms.size());
		int i = 0;
		for (auto it = newdoc.terms.begin(); it != newdoc.terms.end(); ++it)
		{
			newdoc.fastTerms[i] = it->first;
			newdoc.fastCounts[i] = it->second;
			++i;
		}
		docs[k] = newdoc;
		k++;
		percent(lastProcent, (double)k / NUM_OF_DOCS, t0, flagAlreadyPrintTime);
	}
	docs.pop_back();  //удаляю последний, т.к. он некорректно считывается
	
	int totalCategs = 0;
	for (int i = 0; i < categs.size(); ++i)
		if (categs[i].numofthecategory > 0)
			++totalCategs;
	ROBOT_ASSERT(totalCategs == REAL_NUM_OF_CATEGS);
	cout << "categs are read, checked" << endl;
}



void read_test_file(ifstream &fin)
{
	int k = 1;
	double lastProcent = 0;
	clock_t t0 = clock();
	bool flagAlreadyPrintTime = false;
	while (!fin.eof())
	{
		Document newdoc;
		newdoc.numofthedoc = k;
		char ch = ','; // произвольная инициализация
		while (ch != ' ' && !fin.eof())
			fin.get(ch);
		while (ch == ' ' && !fin.eof())
		{
			int feat, value;
			fin >> feat;
			ROBOT_ASSERT_EXT1(feat < NUM_OF_TERMS, feat);
			terms[feat]++;
			fin.get(ch);
			fin >> value;
			newdoc.terms[feat] = value;
			featTestDocs[feat].push_back(k);
			fin.get(ch);
		}
		newdoc.fastTerms.resize(newdoc.terms.size());
		newdoc.fastCounts.resize(newdoc.terms.size());
		int i = 0;
		for (auto it = newdoc.terms.begin(); it != newdoc.terms.end(); ++it)
		{
			newdoc.fastTerms[i] = it->first;
			newdoc.fastCounts[i] = it->second;
			++i;
		}
		testdocs[k] = newdoc;
		k++;
		//cout << k << endl;
		if (k > NUM_OF_TEST_DOCS - 1)
			break;						//TODO
		percent(lastProcent, (double)k / NUM_OF_TEST_DOCS, t0, flagAlreadyPrintTime);
	}
	cout << testdocs[NUM_OF_TEST_DOCS - 1].numofthedoc;
	//testdocs.pop_back();
}


bool reverseCatComp_big(pair <double, int> lhs, pair <double, int> rhs)
{
	return (lhs.first > rhs.first) || ((lhs.first == rhs.first) && (lhs.second > rhs.second));
}


double entropia(double q0, double q1)
{
	if (q0 == 0 || q1 == 0)
		return 0;
	return -q0*log2(q0) - q1*log2(q1);
}

double entropia2(double p, double n)
{
	return entropia(p / (p + n), n / (p + n));
}

int alpha0, alpha1;

bool continuePred(int ii, double entropy, double entropyBarrier)
{
	if (ii < alpha0) return true;
	if (ii > alpha1) return false;
	return entropy >= entropyBarrier;
}


bool compPairGreater (pair <double, int> a, pair <double, int> b)
{ 
	return a.first > b.first; 
}

bool compPairLess(pair <double, int> a, pair <double, int> b)
{ 
	return a.first < b.first; 
}

bool compPairGreaterIntIntSecond (pair <int, int> a, pair <int, int> b)
{ 
	return a.second > b.second; 
}


// предсказание для конкурса
void kaggle_knn_entropy(ofstream &fout, int chosendocs, /*int chosencategs, */ double entropyBarrier)
{
	double sumPrec = 0;
	double lastProcent = 0;
	clock_t t0 = clock();
	bool flagAlreadyPrintTime = false;
#pragma omp parallel for
	for (int i = 0; i < NUM_OF_TEST_DOCS; ++i)
	{
		map <int, int> commonCategs;
		vector <InfoMetrics> docMetrics(NUM_OF_DOCS);
		Document curTestDoc = testdocs[i];
		for (int j = 0; j < NUM_OF_DOCS; ++j)
		{
			if (docs[j].numofthedoc < 0)   // пропускаю несуществующие документы
			{
				docMetrics[j].doc = -1;
				continue;
			}
			docMetrics[j].metrics = curTestDoc.getSimilarity(docs[j]);
			docMetrics[j].doc = j;
		}
		sort(docMetrics.begin(), docMetrics.end(), compare_bigger);
		double totalDocs = 0;
		for (int j = 0; j < chosendocs; ++j)  // chosendocs - число выбранных соседних документов
		{
			int curDoc = docMetrics[j].doc;
			ROBOT_ASSERT(curDoc != -1);
			for (auto it = docs[curDoc].categories.begin(); it != docs[curDoc].categories.end(); ++it)
			{
				if (commonCategs.find(*it) == commonCategs.end())
					totalDocs += categs[*it].numofdocs;
				commonCategs[*it]++; // ищем общие категории
			}
		}

		// multimap - чтобы быстро выбрать категории с самым большим числом элементов
		int tempI = 0;
		vector <pair <double, int> > reverseCommonCategs(commonCategs.size());
		for (auto it = commonCategs.begin(); it != commonCategs.end(); ++it)
		{
			double l = it->second;
			double k = categs[it->first].numofdocs;
			double p = l, n = chosendocs - l, P = k, N = totalDocs - k;
			double ifEntropia = (p + n) / (P + N) * entropia2(p, n) + (P + N - p - n) / (P + N) * entropia2(P - p, N - n);
			double iGain = entropia2(P, N) - ifEntropia;
			if (l / k < k / totalDocs)  // если условная вероятность меньше безусловной
				iGain = 0;
			reverseCommonCategs[tempI++] = make_pair(iGain, it->first);
		}
		sort(reverseCommonCategs.begin(), reverseCommonCategs.end(), reverseCatComp_big);
//#pragma omp critical
		{
			//fout << curTestDoc.numofthedoc << ","; 
			for (int ii = 0; ii < reverseCommonCategs.size() && continuePred(ii, reverseCommonCategs[ii].first,  entropyBarrier); ++ii)
			{
				predictions[i].push_back(reverseCommonCategs[ii].second);
				/*if (ii == 0)
					fout << reverseCommonCategs[ii].second;
				else
					fout << " " << reverseCommonCategs[ii].second;*/
			}
			//++count;
		}
		percent(lastProcent, (double)i / NUM_OF_TEST_DOCS, t0, flagAlreadyPrintTime);
		reverseCommonCategs.clear();
		commonCategs.clear();
		reverseCommonCategs.clear();
	}
}


void kaggle_knn(ifstream & finFreq, ofstream & fout)
{
	ROBOT_ASSERT(terms[2085166] > 0);
	double lastProcent = 0;
	clock_t t0 = clock();
	double t00 = omp_get_wtime();
	bool flagAlreadyPrintTime = false;
	int globaltfidfChoose = 3;
	int globalcloseDocsChoose = 5;
	int done = 0;
#pragma omp parallel for
	for (int j = 1; j < NUM_OF_TEST_DOCS; ++j)  // от 1 до NUM_OF_TEST_DOCS
	{
		int doc = j;
		vector <pair <double, int>> tfidf(testdocs[doc].terms.size());// 1 - tf-idf, 2 - feat
		int k = 0;
		int tfidfChoose = min(globaltfidfChoose, (int)testdocs[doc].terms.size());
		for (auto it = testdocs[doc].terms.begin(); it != testdocs[doc].terms.end(); ++it)
		{
			int curFeat = it->first;
			int sum = 0;
			for (auto it2 = testdocs[doc].terms.begin(); it2 != testdocs[doc].terms.end(); ++it2)
			{
				sum += it2->second;
			}
			ROBOT_ASSERT(sum > 0);
			double tf = (double)it->second / sum;   // здесь общее число слов в документе уникальных или нет?
			ROBOT_ASSERT_EXT1(terms[curFeat] > 0, curFeat);  //added 20/10
			double idf = log2((double)(NUM_OF_TEST_DOCS + NUM_OF_DOCS) / terms[curFeat]); // вынести расчеты за цикл - при формировании terms[curFeat] проделывать
			tfidf[k].first = tf * idf;
			tfidf[k].second= curFeat;
			//cout <<"res = " << tfidf[k].first << ", feat = " << tfidf[k].second << ", it->s = " << it->second << ", t[cF] = " <<  terms[curFeat] << ", sum = " << sum << ", tf = " << tf << ", idf = " << idf << endl;
			++k;
		}
		sort(tfidf.begin(), tfidf.end(), compPairGreater);  
		
		if (tfidfChoose > 1)
			ROBOT_ASSERT(tfidf[0].first >= tfidf[1].first);
		
		vector <pair <double, int>> chosenTfidf(tfidfChoose);
		int count = 0;
		for (int i = 0; i < tfidf.size() && count < tfidfChoose; ++i) 
		{
			if (terms[tfidf[i].second] != 1)
			{				// если этот термин не встречается только в этом документе
				chosenTfidf.push_back(tfidf[i]);
				++count;
			}				
		}
		
		// самый исходный knn
		/*partial_sort(tfidf.begin(), tfidf.begin() + tfidfChoose, tfidf.end(), compPairGreater);
		for (int i = 0; i < tfidfChoose; ++i)
			chosenTfidf[i] = tfidf[i];*/
		
		if (chosenTfidf.empty())
			continue; // нет терминов, по которым искать - искать не будем*/
		
		set <int> neighbourDocs;
		for (int i = 0; i < chosenTfidf.size(); ++i)   // ввести еще 1 степень свободы: количество терминов, по которым должны пересекаться документы, чтобы быть соседями
					// сейчас равно 1.
		{
			int curFeat = chosenTfidf[i].second;
			ROBOT_ASSERT(curFeat < NUM_OF_TERMS);
			if (featDocs[curFeat].empty())
				continue;
			for (auto it = featDocs[curFeat].begin(); it != featDocs[curFeat].end(); ++it)
			{
				//ROBOT_ASSERT(docs[*it].terms.find(curFeat) != docs[*it].terms.end());
				if (*it != doc)
					neighbourDocs.insert(*it);
			}
		}
			
		vector <pair <double, int>> closestDocs(neighbourDocs.size());
		k = 0;
		int closeDocsChoose = min(globalcloseDocsChoose, (int)neighbourDocs.size());
		for (auto it = neighbourDocs.begin(); it != neighbourDocs.end(); ++it)
		{
			ROBOT_ASSERT((*it) < NUM_OF_DOCS);
			closestDocs[k].first = testdocs[doc].getSimilarityKnn(docs[*it]); 
			closestDocs[k].second = *it;
			++k;
		}
		partial_sort(closestDocs.begin(), closestDocs.begin() + closeDocsChoose, closestDocs.end(), compPairLess);
		
		if (closeDocsChoose > 1)
			ROBOT_ASSERT(closestDocs[0].first <= closestDocs[1].first);
		map <int, int> commCategs;  // 1 - categ, 2 - num of repetitions
		for (auto it = closestDocs.begin(); it != closestDocs.begin() + closeDocsChoose; ++it)
		{
			int curDoc = it->second;
			ROBOT_ASSERT_EXT1(curDoc < NUM_OF_DOCS, curDoc);
			for (auto it2 = docs[curDoc].categories.begin(); it2 != docs[curDoc].categories.end(); ++it2)
				commCategs[*it2]++;
		}
		vector <pair<int, int>> sortedCommCategs(commCategs.size());  // 1 - categ, 2 - num of repetitions
		k = 0;
		for (auto it = commCategs.begin(); it != commCategs.end(); ++it)
		{
			sortedCommCategs[k] = make_pair(it->first, it->second);
			++k;
		}
		int categsChoose = min(3, (int)commCategs.size());
		partial_sort(sortedCommCategs.begin(), sortedCommCategs.begin() + categsChoose, sortedCommCategs.end(), compPairGreaterIntIntSecond);
		if (categsChoose > 1)
			ROBOT_ASSERT(sortedCommCategs[0].second >= sortedCommCategs[1].second);
		int lastPredcategRep;
		if (categsChoose > 0)
			lastPredcategRep = sortedCommCategs[categsChoose - 1].second;
		else 
			continue;  // ничего не предсказали
			//ROBOT_ASSERT_EXT1(false, doc); // ничего не предсказываем 
		// почему переходит по ветке else? потому что случается, что в документе самое высокое tf-idf показывают те термины, которые встречаются только в нем
		// например, документ 1921947, информация о нем в log6november1.txt
		// чтобы это исправить, запрещу брать в качестве определяющих термины, которые содержатся только в одном документе
		// т.к. они не показательные
		
		
		// 1788509 (doc), 1788511 (line) - документ, в котором только 1 термин, уникальный для него -> вынуждены делать continue
		// sed '1788511q;d' train.csv - команда для того, чтобы найти строчку
		#pragma omp critical
		{
			//fout << doc << ",";
			for (int i = 0; i < categsChoose; ++i)
				predictions[doc].push_back(sortedCommCategs[i].first);
				//fout << sortedCommCategs[i].first << " ";
				//predictedDocs[sortedCommCategs[i].first].push_back(doc);
			for (int i = categsChoose; i < sortedCommCategs.size(); ++i)
				if (lastPredcategRep == sortedCommCategs[i].second)
					predictions[doc].push_back(sortedCommCategs[i].first);
					//fout << sortedCommCategs[i].first << " ";
				//predictedDocs[sortedCommCategs[i].first].push_back(doc);
			//fout << endl;
			++done;
			if (done == 1)
				cout << "num threads = " << omp_get_num_threads() << endl;
			if (done % 1000 == 0)
			{
				double curTime = (double)(omp_get_wtime() - t00) / 3600;
				cout << "done = " << done << ", time = " << curTime << ", remain = " << curTime * (NUM_OF_TEST_DOCS - done) / done << endl;
			}
			percent(lastProcent, (double)done / NUM_OF_TEST_DOCS / omp_get_num_threads(), t0, flagAlreadyPrintTime);
		}
	}
	for (int i = 1; i < NUM_OF_TEST_DOCS; ++i)  // От 1 т.к. первый документ не заполнен ничем
	{
		fout << i << ","; 
		for (int ii = 0; ii < predictions[i].size(); ++ii)
		{
			if (ii == 0)
				fout << predictions[i][ii];
			else
				fout << " " << predictions[i][ii];
		}
		fout << endl;
	}
}

void sort(valarray <EntropiaPair>& x, valarray<int>& ind)
{
    int n = x.size();
    vector <pair<int, EntropiaPair> > order(n);
    for (int i = 0; i < n; i++) 
		order[i] = make_pair(i, x[i]);
    struct ordering 
	{
        bool operator ()(pair <int, EntropiaPair> const& a, pair<int, EntropiaPair>  const& b) 
		{
			 return (a.second.categ < b.second.categ) || (a.second.categ == b.second.categ && a.second.entropia > b.second.entropia) ;
		}
    };
	struct orderingByEntropy
	{
		 bool operator ()(pair <int, EntropiaPair> const& a, pair<int, EntropiaPair>  const& b) 
		{
			 return a.second.entropia > b.second.entropia;
		}
	};
	struct orderingByDoc
	{
		bool operator ()(pair <int, EntropiaPair> const& a, pair<int, EntropiaPair>  const& b) 
		{
			 return (a.second.doc < b.second.doc) || (a.second.doc == b.second.doc && a.second.entropia > b.second.entropia);
		}
	};
     sort(order.begin(), order.end(), ordering());
     ind.resize(n);
     for (int i = 0; i < n; i++) 
		 ind[i] = order[i].first;
}

valarray <EntropiaPair> totalEntropiaTest(NUM_OF_PAIRS_TEST);

void readFileTotalEntropiaTest(FILE * fin)
{
	fread(&totalEntropiaTest[0], sizeof(EntropiaPair), NUM_OF_PAIRS_TEST, fin); // одной командой fread
	valarray <int> ind(NUM_OF_PAIRS_TEST);
	sort(totalEntropiaTest, ind); 	// следить, какой сорт используетсЯ!			
	valarray <EntropiaPair> totalEntropiaTestSorted(NUM_OF_PAIRS_TEST);
	for (int i = 0; i < totalEntropiaTest.size(); ++i)
		totalEntropiaTestSorted[i] = totalEntropiaTest[ind[i]];
	totalEntropiaTest = totalEntropiaTestSorted;
	/*for (int i = 0; i < 10; ++i)
		cout << totalEntropia[i].doc << " " << totalEntropia[i].categ << " " << totalEntropia[i].entropia << endl;*/
}


void kaggleEntrBestR(ifstream & finBestR, ofstream & fout, ofstream &foutMistakes)  // sort - по категориям, внутри по энтропии
{
	double lastProcent = 0;
	clock_t t0 = clock();
	bool flagAlreadyPrintTime = false;
	int j = 0;
	int tempNum;
	predictions.clear();
	vector <int> r(NUM_OF_CATEGS);
	for (int i = 0; i < NUM_OF_CATEGS; ++i)
	{
		finBestR >> tempNum >> r[tempNum];
		if (r[tempNum] > 0)  // нововведение 1
			r[tempNum] = ceil((double)r[tempNum] * NUM_OF_TEST_DOCS / NUM_OF_DOCS); // масштабирование границ
		r[24177] = 0;
	}
	cout << "r[1] = " << r[1] << endl;
	cout << "r[159007] = " << r[159007] << endl;
	ROBOT_ASSERT(r[499999] == -1000000000);
	set <int> watchedCategs;
	while (j < NUM_OF_PAIRS_TEST)
	{
		int count = 0;
		int categ = totalEntropiaTest[j].categ;
		ROBOT_ASSERT(watchedCategs.find(categ) == watchedCategs.end());
		watchedCategs.insert(categ);
		while (j < NUM_OF_PAIRS_TEST && totalEntropiaTest[j].categ == categ && count < r[categ])
		{
			/*if (count > 0)
				ROBOT_ASSERT(totalEntropiaTest[j - 1].entropia >= totalEntropiaTest[j].entropia);*/
			predictions[totalEntropiaTest[j].doc].push_back(categ);
			if (totalEntropiaTest[j].doc == 0)
				foutMistakes << "doc = " << totalEntropiaTest[j].doc << ", categ = " << totalEntropiaTest[j].categ << ", entr = " << totalEntropiaTest[j].entropia << endl;
			++j;
			++count;
		}
		while (totalEntropiaTest[j].categ == categ)
			++j;
		percent(lastProcent, (double)j / NUM_OF_PAIRS_TEST, t0, flagAlreadyPrintTime);
	}
	for (int i = 1; i < NUM_OF_TEST_DOCS; ++i)  // От 1 т.к. первый документ не заполнен ничем
	{
		fout << i << ","; 
		for (int ii = 0; ii < predictions[i].size(); ++ii)
		{
			if (ii == 0)
				fout << predictions[i][ii];
			else
				fout << " " << predictions[i][ii];
		}
		fout << endl;
	}
}

int main()
{
	srand(1);
	ifstream fin, fintest, finFreq;
	fin.open("train.csv");
	fintest.open("test.csv");
	finFreq.open("frequencyGlobal.txt");
	

	for (int i = 0; i < NUM_OF_DOCS; ++i)
	{
		docs[i].numofthedoc = -1000000000;
	}

	for (int i = 0; i < NUM_OF_CATEGS; ++i)
	{
		categs[i].numofthecategory = -1000000000;
	}

	//read_all_docs(fin);	
	//read_all_docs_without_catterms(fin);
	cout << "finish processing train file" << endl;
	
	FILE * finEntr;
	if((finEntr = fopen("entropiaForTest.bin", "rb"))== NULL) 
	{
		printf("error opening \n");
		exit(1);
	}
	readFileTotalEntropiaTest(finEntr);
	
	
	terms.assign(NUM_OF_TERMS, 0);
	for (int i = 0; i < REAL_NUM_OF_TERMS; ++i)
	{
		int k, df;
		finFreq >> k >> df;
		terms[k] = df;
	}

	//read_test_file(fintest);
	
/*	for (int i = 0; i < categories[24177].size(); ++i)
	{
		cout << categories[24177]
	}*/

	cout << "success" << endl;
	
	ifstream finBestR;
	finBestR.open("bestRforcategs.txt");

	// вывод информации о документах в файл
	ofstream fout;
	fout.open("KaggleEntr301115-1(except24177).csv");
	fout << "Id,Predicted"<< endl;
	
	
	ofstream foutMistakes;
	foutMistakes.open("mistakes3011.txt");
	/*int doc = 452167;	
	int j = 0;
	while (j < NUM_OF_PAIRS_TEST && totalEntropiaTest[j].doc != doc)
		++j;
	while (j < NUM_OF_PAIRS_TEST && totalEntropiaTest[j].doc == doc)
	{
		foutMistakes << "doc = " << doc << ", categ = " << totalEntropiaTest[j].categ << ", entr = " << totalEntropiaTest[j].entropia << endl;
		++j;
	}*/
	
	double entropyBarrier = 5e-06;
	int chosendocs = 50;
	alpha0 = 3;
	alpha1 = 3;
	//kaggle_knn_entropy(fout, chosendocs, entropyBarrier);
	
	//kaggle_knn(finFreq, fout);
	
	kaggleEntrBestR(finBestR, fout, foutMistakes);
	
	//fout.close();


	return 0;
}