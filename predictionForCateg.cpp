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
#include <sys/time.h>
#include "macros.cpp"
#define TIMER
#include "tictoc.c"
//#include <boost/math/distributions/hypergeometric.hpp>


using namespace std;

const int NUM_OF_DOCS = 2365436;  //  //742tk 
const int NUM_OF_CATEGS = 500000; // 
const int REAL_NUM_OF_CATEGS = 325056;
const int NUM_OF_TERMS = 2085167;  // если считать terms, которых вообще нет, удобная константа для индексации
const int REAL_NUM_OF_TERMS = 1617899;
const int NUM_OF_PAIRS = 179682414;    // в прошлой версии почему-то был на 1 меньше??
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

struct EntropiaPair
{
	int doc;
	int categ;
	double entropia;
};

struct Frequency
{
	int term, df;
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

bool compare_bigger(InfoMetrics lhs, InfoMetrics rhs)
{
	return lhs.metrics > rhs.metrics;
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


valarray <EntropiaPair> totalEntropia(NUM_OF_PAIRS);
valarray <Frequency> totalFrequency(REAL_NUM_OF_TERMS);
vector <Document> docs(NUM_OF_DOCS + 1);
vector <Category> categs(NUM_OF_CATEGS);
vector <int> terms(NUM_OF_TERMS);
map <int, list <int>> featDocs;

vector <int> truepos(NUM_OF_CATEGS);
vector <int> falsepos(NUM_OF_CATEGS);
vector <int> r(NUM_OF_CATEGS);



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

bool continuePred(int count, double entropy, double entropyBarrier, int alpha0, int alpha1)
{
	if (count < alpha0) return true;
	if (count > alpha1) return false;
	return entropy >= entropyBarrier;
}

double makePredictionForCateg(double entropyBarrier, int alpha0, int alpha1)
{
	double MaF = 0, MaR = 0, MaP = 0;
	double lastProcent = 0;
	clock_t t0 = clock();
	bool flagAlreadyPrintTime = false;
	int j = 0;
	int checkcategs = 0;
	while (j < NUM_OF_PAIRS)
	{
		int categ = totalEntropia[j].categ;
		int truepos = 0, falsepos = 0;
		int count = 0;
	//	double entropia = totalEntropia[j].entropia;
		while (j < NUM_OF_PAIRS && totalEntropia[j].categ == categ && continuePred(count, totalEntropia[j].entropia, entropyBarrier, alpha0, alpha1))
		{
			if (count > 0)
				ROBOT_ASSERT(totalEntropia[j - 1].entropia >= totalEntropia[j].entropia);
			int predDoc = totalEntropia[j].doc;
			if (docs[predDoc].categories.find(categ) != docs[predDoc].categories.end())
				truepos++;
			else
				falsepos++;
			++j;
			++count;
		}
		while (totalEntropia[j].categ == categ)
			++j;
		if (j < NUM_OF_PAIRS)
			ROBOT_ASSERT(categ < totalEntropia[j].categ)
		checkcategs++;
		MaP += (double)truepos / count;
		MaR += (double)truepos / categs[categ].numofdocs;
		percent(lastProcent, (double)j / NUM_OF_PAIRS, t0, flagAlreadyPrintTime);
		//truepos = 0;
	}
	ROBOT_ASSERT_EXT2(checkcategs <= REAL_NUM_OF_CATEGS, checkcategs, REAL_NUM_OF_CATEGS);
	MaR /= REAL_NUM_OF_CATEGS;
	MaP /= REAL_NUM_OF_CATEGS;
	MaF = 2 * MaP * MaR / (MaP + MaR);
	cout << "MaR = " << MaR << ", MaP = " << MaP << ", MaF = " << MaF << endl;
	return MaF;
}

void findMistakes(int chosen, ofstream & fout)
{
	//for (int i = 0; i < chosen; ++i)
	double entropy = 1;
	int i = 0;
	while (entropy > 0.1)
	{
		entropy = totalEntropia[i].entropia;
		int doc =  totalEntropia[i].doc;
		int categ = totalEntropia[i].categ;
		bool belong = docs[doc].categories.find(categ) != docs[doc].categories.end();
		fout << "entr = " << entropy << ", doc = " << doc << ", categ = " << categ << 
		   ", in train =" << belong << endl;
		++i;
	}
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


void checkEntropia(int doc, int categ, int chosendocs, ofstream & fout)
{
	fout << "terms of the doc: " << endl;
	for (auto it = docs[doc].terms.begin(); it != docs[doc].terms.end(); ++it) //вывели все термины в файл посмотреть
	{
		fout << "feat:" << it->first << ", value:" << it->second << endl;
	}
	map <int, int> commonCategs;
	vector <InfoMetrics> docMetrics(NUM_OF_DOCS);
	for (int j = 0; j < NUM_OF_DOCS; ++j)
	{
		if (docs[j].numofthedoc < 0 || j == doc)   // пропускаю несуществующие документы
		{
			docMetrics[j].doc = -1;
			continue;
		}
		docMetrics[j].metrics = docs[doc].getSimilarity(docs[j]);  
		docMetrics[j].doc = j;
	}
	fout << "the closest chosendocs: " << endl;
	sort(docMetrics.begin(), docMetrics.end(), compare_bigger);
	for (int j = 0; j < chosendocs; ++j)
	{
		fout << docMetrics[j].doc << ", metrics = " << docMetrics[j].metrics << endl;
	}
	double totalDocs = 0;
	for (int j = 0; j < chosendocs; ++j)  // chosendocs - число выбранных соседних документов
	{
		int curDoc = docMetrics[j].doc;
		fout << "categories of the doc " << curDoc << ": ";
		for (auto it = docs[curDoc].categories.begin(); it != docs[curDoc].categories.end(); ++it)
		{
			fout << *it << ", ";
		}
		fout << endl;
		ROBOT_ASSERT(curDoc != -1);
		for (auto it = docs[curDoc].categories.begin(); it != docs[curDoc].categories.end(); ++it)
		{
			if (commonCategs.find(*it) == commonCategs.end())  // если еще не добавлЯли ничего
				totalDocs += categs[*it].numofdocs; 
			commonCategs[*it]++; // ищем общие категории
		}
	}
	fout << "commonCategs from chosendocs" << endl;
	for (auto it = commonCategs.begin(); it != commonCategs.end(); ++it)
	{
		fout << "categ = " << it->first << ", howMuchDocsHave = " << it->second << endl;
	}
	
	for (auto it = commonCategs.begin(); it != commonCategs.end(); ++it)
	{
		double l = it->second; //в скольких документах окрестности документа встретилась категориЯ 
		double k = categs[it->first].numofdocs;  // сколько в категории документов вообще
		double p = l, n = chosendocs - l, P = k, N = totalDocs - k;

		// n-документы, в которых не встретилась категориЯ
		// N-документы, которые не попали в категорию (по обучающему множеству)
		
		double ifEntropia = (p + n) / (P + N) * entropia2(p, n) + (P + N - p - n) / (P + N) * entropia2(P - p, N - n);
		double iGain = entropia2(P, N) - ifEntropia;
		if (l / k < k / totalDocs)  // если условнаЯ вероЯтность меньше безусловной
			iGain = 0;
		bool belong = docs[doc].categories.find(it->first) != docs[doc].categories.end();
		fout << "categ = " << it->first << ", iGain = " << iGain << ", belong = " << belong << endl;
	}
}

/*void makePredictionForCategKaggle(double entropyBarrier, int chosendocs, int alpha0, int alpha1, ofstream & fout)
{
	double lastProcent = 0;
	clock_t t0 = clock();
	bool flagAlreadyPrintTime = false;
	int j = 0;
	while (j < NUM_OF_PAIRS)
	{
		fout << totalEntropia[j].categ << ",";
		while (totalEntropia[j].categ == categ && continuePred(count, totalEntropia[j].entropia, entropyBarrier, alpha0, alpha1)  && j < NUM_OF_PAIRS)
		{
			fout << totalEntropia[j].doc << " ";
		}
		fout << "/n";
		percent(lastProcent, (double)j / NUM_OF_PAIRS, t0, flagAlreadyPrintTime);
	}
}
*/

void processEntropyFile()
{
	double MaR = 0, MaP = 0;
	double lastProcent = 0;
	clock_t t0 = clock();
	bool flagAlreadyPrintTime = false;
	
	int j = 0;
	while (j < NUM_OF_PAIRS)
	{
		int categ = totalEntropia[j].categ;
		int truepos = 0;
		while (j < NUM_OF_PAIRS && totalEntropia[j].categ == categ)
		{
			int predDoc = totalEntropia[j].doc;
			if (docs[predDoc].categories.find(categ) != docs[predDoc].categories.end())
				truepos++;
			++j;
		}
		MaR += (double)truepos / categs[categ].numofdocs;
		percent(lastProcent, (double)j / NUM_OF_PAIRS, t0, flagAlreadyPrintTime);
	}
	double MaRwrong = MaR / NUM_OF_CATEGS;
	MaR /= REAL_NUM_OF_CATEGS;
	cout << "Ideal MaR = " << MaR << ", wrong MaR = " << MaRwrong << endl;
}

void bestRforCategs(/*ofstream & fout*/)  // sort - по категориям, а внутри по энтропии - название ordering
{
	int j = 0;
	double lastProcent = 0;
	clock_t t0 = clock();
	bool flagAlreadyPrintTime = false;
	set <int> watchedCategs;
	while (j < NUM_OF_PAIRS)
	{
		int categ = totalEntropia[j].categ;
		int tp = 0, fp = 0;
		int count = 0;
		double maxMaf = 0;
		r[categ] = 0;
		truepos[categ] = 0;
		falsepos[categ] = 0;
		int maxcount, maxtp, maxfp;
		ROBOT_ASSERT(watchedCategs.find(categ) == watchedCategs.end());
		watchedCategs.insert(categ);
	//	double entropia = totalEntropia[j].entropia;
		while (j < NUM_OF_PAIRS && totalEntropia[j].categ == categ)
		{
			double curMaf = 0;
			if (count > 0)
				ROBOT_ASSERT(totalEntropia[j - 1].entropia >= totalEntropia[j].entropia);
			int predDoc = totalEntropia[j].doc;
			if (docs[predDoc].categories.find(categ) != docs[predDoc].categories.end())
				tp++;
			else
				fp++;
			++j;
			++count;
			if (tp != 0)
			{
				double curMaP = (double)tp / count;
				double curMaR = (double)tp / categs[categ].numofdocs;
				curMaf = 2 * curMaP * curMaR / (curMaR + curMaP);
			}
			if (curMaf > maxMaf)
			{
				maxMaf = curMaf;
				maxcount = count;
				maxtp = tp;
				maxfp = fp;
			}
		}
	//	ROBOT_ASSERT(count <= categs[categ].numofdocs);
		if (maxMaf != 0)
		{
			r[categ] = maxcount;
			truepos[categ] = maxtp;
			falsepos[categ] = maxfp;
		}
		if (j < NUM_OF_PAIRS)
			ROBOT_ASSERT(categ < totalEntropia[j].categ)
		percent(lastProcent, (double)j / NUM_OF_PAIRS, t0, flagAlreadyPrintTime);
	}
	
	double MaP = 0, MaR = 0;
	for (int i = 0; i < NUM_OF_CATEGS; ++i)
	{
		if (truepos[i] < 0)
			continue;
		if (r[i] != 0)
			MaP += (double)truepos[i] / r[i];
		MaR += (double)truepos[i] / categs[i].numofdocs;
	}
	MaP /= REAL_NUM_OF_CATEGS;
	MaR /= REAL_NUM_OF_CATEGS;
	double MaF = 2 * MaP * MaR / (MaP + MaR);
	cout << "MaR = " << MaR << ", MaP = " << MaP<< ", commonMaF = " << MaF << endl;
	int count = 0;
	while (true)   //достаточное число шагов, чтобы MaF перестал расти
	{
		int j = 0;
		double lastProcent = 0;
		clock_t t0 = clock();
		bool flagAlreadyPrintTime = false;
		while (j < NUM_OF_PAIRS)
		{
			int categ = totalEntropia[j].categ;
			int tp = 0, fp = 0;
			int count = 0;
		//	double entropia = totalEntropia[j].entropia;
			while (j < NUM_OF_PAIRS && totalEntropia[j].categ == categ)
			{
				double curMapOther = MaP - 1.0 / REAL_NUM_OF_CATEGS * truepos[categ] / r[categ];
				double curMarOther = MaR - 1.0 / REAL_NUM_OF_CATEGS * truepos[categ] / categs[categ].numofdocs;
				double curMaP = curMapOther;
				double curMaR = curMarOther;
				double curMaf = 0;
				int predDoc = totalEntropia[j].doc;
				if (docs[predDoc].categories.find(categ) != docs[predDoc].categories.end())
					tp++;
				else
					fp++;
				++j;
				++count;
				if (tp != 0)
				{
					curMaP = (double)tp / count / REAL_NUM_OF_CATEGS + curMapOther;
					curMaR = (double)tp / categs[categ].numofdocs / REAL_NUM_OF_CATEGS + curMarOther;
				}
				curMaf = 2 * curMaP * curMaR / (curMaR + curMaP);
				if (curMaf > MaF)
				{
					MaF = curMaf;
					MaP = curMaP;
					MaR = curMaR;
					r[categ] = count;
					truepos[categ] = tp;
					falsepos[categ] = fp;
				}
			}
			if (j < NUM_OF_PAIRS)
				ROBOT_ASSERT(categ < totalEntropia[j].categ)
			percent(lastProcent, (double)j / NUM_OF_PAIRS, t0, flagAlreadyPrintTime);
		}
		cout << "MaR = " << MaR << ", MaP = " << MaP<< ", commonMaF = " << MaF << endl;
		++count;
	}
	/*for (int i = 0; i < NUM_OF_CATEGS; ++i)
		fout << i << " " << r[i] << endl;*/
	
}

void getFrequencyLocal(ofstream & fout)
{
	terms.assign(NUM_OF_TERMS, 0);
	int j = 0;
	double lastProcent = 0;
	clock_t t0 = clock();
	bool flagAlreadyPrintTime = false;
	while (j < NUM_OF_PAIRS)
	{
		int doc = totalEntropia[j].doc;
		for (auto it = docs[doc].terms.begin(); it != docs[doc].terms.end(); ++it)
			terms[it->first]++;
		while (j < NUM_OF_PAIRS && totalEntropia[j].doc == doc)
			++j;
		if (j < NUM_OF_PAIRS)
			ROBOT_ASSERT(doc < totalEntropia[j].doc);
		percent(lastProcent, (double)j / NUM_OF_PAIRS, t0, flagAlreadyPrintTime);
	}
	for (int i = 0; i < terms.size(); ++i)
		if (terms[i])
			fout << i << " " << terms[i] << endl;
}

void getFrequency(ofstream & fout)
{
	terms.assign(NUM_OF_TERMS, 0);
	int j = 0;
	double lastProcent = 0;
	clock_t t0 = clock();
	bool flagAlreadyPrintTime = false;
	for (int j = 0; j < NUM_OF_DOCS; ++j)
	{
		int doc = j;
		for (auto it = docs[doc].terms.begin(); it != docs[doc].terms.end(); ++it)
			terms[it->first]++;
		percent(lastProcent, (double)j / NUM_OF_PAIRS, t0, flagAlreadyPrintTime);
	}
	for (int i = 0; i < terms.size(); ++i)
	{
		if (terms[i])
			fout << i << " " << terms[i] << endl;
	}
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

void knnBenchmark(ifstream & fin) 
 {
	//fread(&totalFrequency[0], sizeof(Frequency), REAL_NUM_OF_TERMS, fin);
	// cout << totalFrequency[3].term << " " << totalFrequency[3].df;  //контроль
	terms.assign(NUM_OF_TERMS, 0);
	for (int i = 0; i < REAL_NUM_OF_TERMS; ++i)
	{
		int k, df;
		fin >> k >> df;
		terms[k] = df;
	}
	ROBOT_ASSERT(terms[2085166] > 0);
	double lastProcent = 0;
	clock_t t0 = clock();
	double t00 = omp_get_wtime();
	bool flagAlreadyPrintTime = false;
	int globaltfidfChoose = 3;
	int globalcloseDocsChoose = 5;
	vector <vector <int> > predictedDocs(NUM_OF_CATEGS); // categs, to each a vector of docs is predicted
	int done = 0;
#pragma omp parallel for
	for (int j = 0; j < NUM_OF_DOCS; ++j)  // от 0 до NUM_OF_DOCS
	{
		int doc = j;
		vector <pair <double, int>> tfidf(docs[doc].terms.size());// 1 - tf-idf, 2 - feat
		int k = 0;
		int tfidfChoose = min(globaltfidfChoose, (int)docs[doc].terms.size());
		for (auto it = docs[doc].terms.begin(); it != docs[doc].terms.end(); ++it)
		{
			int curFeat = it->first;
			int sum = 0;
			for (auto it2 = docs[doc].terms.begin(); it2 != docs[doc].terms.end(); ++it2)
			{
				sum += it2->second;
			}
			ROBOT_ASSERT(sum > 0);
			double tf = (double)it->second / sum;   // здесь общее число слов в документе уникальных или нет?
			ROBOT_ASSERT_EXT1(terms[curFeat] > 0, curFeat);  //added 20/10
			double idf = log2((double)NUM_OF_DOCS / terms[curFeat]); // вынести расчеты за цикл - при формировании terms[curFeat] проделывать
			tfidf[k].first = tf * idf;
			tfidf[k].second= curFeat;
			//cout <<"res = " << tfidf[k].first << ", feat = " << tfidf[k].second << ", it->s = " << it->second << ", t[cF] = " <<  terms[curFeat] << ", sum = " << sum << ", tf = " << tf << ", idf = " << idf << endl;
			++k;
		}
		sort(tfidf.begin(), tfidf.end(), compPairGreater);  // можно попробовать partial_sort первые 100-1000, например, и сравнить результаты
		
		if (tfidfChoose > 1)
			ROBOT_ASSERT(tfidf[0].first >= tfidf[1].first);
		
		vector <pair <double, int>> chosenTfidf;
		int count = 0;
		for (int i = 0; i < tfidf.size() && count < tfidfChoose; ++i) 
		{
			if (terms[tfidf[i].second] != 1)
			{				// если этот термин не встречается только в этом документе
				chosenTfidf.push_back(tfidf[i]);
				++count;
			}				
		}
		
		if (chosenTfidf.empty())
			continue; // нет терминов, по которым искать - искать не будем
		
		/*cout << "best tf-idf terms" << endl;
		for (int i = 0; i < chosenTfidf.size(); ++i)
			cout << chosenTfidf[i].first << " " << chosenTfidf[i].second << endl;*/
		
		set <int> neighbourDocs;
		for (int i = 0; i < chosenTfidf.size(); ++i)   // ввести еще 1 степень свободы: количество терминов, по которым должны пересекаться документы, чтобы быть соседями
					// сейчас равно 1.
		{
			int curFeat = chosenTfidf[i].second;
			ROBOT_ASSERT(curFeat < NUM_OF_TERMS);
			for (auto it = featDocs[curFeat].begin(); it != featDocs[curFeat].end(); ++it)
			{
				//ROBOT_ASSERT(docs[*it].terms.find(curFeat) != docs[*it].terms.end());
				if (*it != doc)
					neighbourDocs.insert(*it);
			}
		}
		/*cout << "terms of cur doc" << endl;
		for (auto it = docs[j].terms.begin(); it != docs[j].terms.end(); ++it)
			cout << it->first << " ";
		cout << "terms of neighb docs" << endl;
		for (auto it2 = neighbourDocs.begin(); it2 != neighbourDocs.end(); ++it2)
		{
			cout << "doc: " << *it2 << ". ";
			for (auto it = docs[*it2].terms.begin(); it != docs[*it2].terms.end(); ++it)
				cout << it->first << " ";
			cout << endl;
		}*/
			
		vector <pair <double, int>> closestDocs(neighbourDocs.size());
		k = 0;
		int closeDocsChoose = min(globalcloseDocsChoose, (int)neighbourDocs.size());
		for (auto it = neighbourDocs.begin(); it != neighbourDocs.end(); ++it)
		{
			ROBOT_ASSERT((*it) < NUM_OF_DOCS);
			closestDocs[k].first = docs[doc].getSimilarityKnn(docs[*it]); 
			closestDocs[k].second = *it;
			++k;
		}
		partial_sort(closestDocs.begin(), closestDocs.begin() + closeDocsChoose, closestDocs.end(), compPairLess);
		
		/*cout << "terms of cur doc" << endl;
		for (auto it = docs[j].terms.begin(); it != docs[j].terms.end(); ++it)
			cout << it->first << " ";
		cout << endl;
		for (auto it2 = closestDocs.begin(); it2 != closestDocs.begin() + closeDocsChoose; ++it2)
		{
			cout << "doc: " << it2->second << ": " << "simil = " << it2->first << endl;
			for (auto it = docs[it2->second].terms.begin(); it != docs[it2->second].terms.end(); ++it)
				cout << it->first << " ";
			cout << endl;
		}*/
		
		if (closeDocsChoose > 1)
			ROBOT_ASSERT(closestDocs[0].first <= closestDocs[1].first);
		map <int, int> commCategs;  // 1 - categ, 2 - num of repetitions
		for (auto it = closestDocs.begin(); it != closestDocs.begin() + closeDocsChoose; ++it)
		{
			int curDoc = it->second;
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
		else ROBOT_ASSERT_EXT1(false, doc); // ничего не предсказываем 
		// почему переходит по ветке else? потому что случается, что в документе самое высокое tf-idf показывают те термины, которые встречаются только в нем
		// например, документ 1921947, информация о нем в log6november1.txt
		// чтобы это исправить, запрещу брать в качестве определяющих термины, которые содержатся только в одном документе
		// т.к. они не показательные
		
		
		// 1788509 (doc), 1788511 (line) - документ, в котором только 1 термин, уникальный для него -> вынуждены делать continue
		// sed '1788511q;d' train.csv - команда для того, чтобы найти строчку
		#pragma omp critical
		{
			for (int i = 0; i < categsChoose; ++i)
				predictedDocs[sortedCommCategs[i].first].push_back(j);
			
			for (int i = categsChoose; i < sortedCommCategs.size(); ++i)
				if (lastPredcategRep == sortedCommCategs[i].second)
					predictedDocs[sortedCommCategs[i].first].push_back(j);
			++done;
			if (done == 1)
				cout << "num threads = " << omp_get_num_threads() << endl;
			if (done % 1000 == 0)
			{
				double curTime = (double)(omp_get_wtime() - t00) / 3600;
				cout << "done = " << done << ", time = " << curTime << ", remain = " << curTime * (NUM_OF_DOCS - done) / done << endl;
			}
			percent(lastProcent, (double)done / NUM_OF_DOCS / omp_get_num_threads(), t0, flagAlreadyPrintTime);
		}
		/*cout << "numofthedoc = " << j << endl;
		cout << "\n predicted by knn: " << endl;
		int countCommonKnn = 0, countCommonEntr = 0;
		for (int i = 0; i < categsChoose; ++i)
		{
			cout << sortedCommCategs[i].first << " ";
			if (docs[j].categories.find(sortedCommCategs[i].first) != docs[j].categories.end())
				countCommonKnn++;
		}
		for (int i = categsChoose; i < sortedCommCategs.size(); ++i)
				if (lastPredcategRep == sortedCommCategs[i].second)
				{
					cout << sortedCommCategs[i].first << " ";
					if (docs[j].categories.find(sortedCommCategs[i].first) != docs[j].categories.end())
						countCommonKnn++;	
				}
		cout << "\n predicted by entropia: " << endl;
		k = 0;
		while (k < NUM_OF_PAIRS && totalEntropia[k].doc != j)
			++k;
		cout << "numofthedoc from entr (check) = " << totalEntropia[k].doc << endl;
		while (k < NUM_OF_PAIRS && totalEntropia[k].doc == j)
		{
			cout << totalEntropia[k].categ << " ";
			if (docs[j].categories.find(totalEntropia[k].categ) != docs[j].categories.end())
				countCommonEntr++;
			++k;
		}
		ROBOT_ASSERT(totalEntropia[k].doc > j);
		cout << "\n real categories: " << endl;
		for (auto it = docs[j].categories.begin(); it != docs[j].categories.end(); ++it)
			cout << *it << " ";
		cout << "kNN = " << countCommonKnn << ", entr = " << countCommonEntr << endl;
		//exit(-1); */ 
	}
	double MaR = 0, MaP = 0;
	int countCategs = 0;
	for (int i = 0; i < NUM_OF_CATEGS; ++i)
	{
		int tp = 0, fp = 0;
		int categ = i;
		if (!predictedDocs[i].empty())
		{
			for (int j = 0; j < predictedDocs[i].size(); ++j)
			{
				int predDoc = predictedDocs[i][j];
				if (docs[predDoc].categories.find(categ) != docs[predDoc].categories.end())
					tp++;
				else
					fp++;
			}
			countCategs++;
		}
		else continue;
		if (tp != 0)
		{
			MaP += (double)tp / predictedDocs[i].size();  
			MaR += (double)tp / categs[categ].numofdocs;
		}
		/*if (i % 10000)
			cout << "MaR = " << MaR << ", MaP = " << MaP << endl;*/
	}
	
	cout << "categs in set = " << countCategs << endl;
	double MaP2 = MaP / countCategs;
	double MaR2 = MaR / countCategs;
	double MaF2 = 2 * MaP2 * MaR2 / (MaP2 + MaR2);
	cout << "MaR2 = " << MaR2 << ", MaP2 = " << MaP2 << ", commonMaF2 = " << MaF2 << endl;
	
	MaP /= REAL_NUM_OF_CATEGS;
	MaR /= REAL_NUM_OF_CATEGS;
	double MaF = 2 * MaP * MaR / (MaP + MaR);
	cout << "MaR = " << MaR << ", MaP = " << MaP << ", commonMaF = " << MaF << endl;
} 


// чтение большого файла totalEntropia
void readFileTotalEntropia(FILE * fin)
{
	fread(&totalEntropia[0], sizeof(EntropiaPair), NUM_OF_PAIRS, fin); // одной командой fread
	valarray <int> ind(NUM_OF_PAIRS);
	sort(totalEntropia, ind); 	// следить, какой сорт используетсЯ!			
	valarray <EntropiaPair> totalEntropiaSorted(NUM_OF_PAIRS);
	for (int i = 0; i < totalEntropia.size(); ++i)
		totalEntropiaSorted[i] = totalEntropia[ind[i]];
	totalEntropia = totalEntropiaSorted;
	/*for (int i = 0; i < 10; ++i)
		cout << totalEntropia[i].doc << " " << totalEntropia[i].categ << " " << totalEntropia[i].entropia << endl;*/
}

double entropyBarrier;
int chosendocs, alpha0, alpha1;

void testParameters()
{
	for (int i = 0; i < 10; ++i)
		cout << totalEntropia[i].doc << " " << totalEntropia[i].categ << " " <<totalEntropia[i].entropia << endl;
	while (true)  // TODO: обработка ошибок
	{
		cout << "Enter entropy barrier: ";
		cin >> entropyBarrier;
		cout << "Enter alpha0: ";
		cin >> alpha0;
		cout << "Enter alpha1: ";
		cin >> alpha1;
		makePredictionForCateg(entropyBarrier, alpha0, alpha1);
	}
}


// алгоритм старый, но качество теперь удобно считать, т.к. есть информация
// обо всех энтропиях

int main()
{
	//cout << numOfPairs << endl;
	srand(1);
	FILE * fin;
	if((fin = fopen("entropiaForTrain.bin", "rb"))== NULL) 
	{
		printf("error opening \n");
		exit(1);
	}
  //  tic(3);
	readFileTotalEntropia(fin);
   // toc(3);
   // printAllTimers(1);
	
	for (int i = 0; i < NUM_OF_DOCS; ++i)
	{
		docs[i].numofthedoc = -1000000000;
	}

	for (int i = 0; i < NUM_OF_CATEGS; ++i)
	{
		categs[i].numofthecategory = -1000000000;
		truepos[i] = -1000000000;
		falsepos[i] = -1000000000;
		r[i] = -1000000000;
	}
	
	ifstream finTrainFile;
	finTrainFile.open("train.csv");
	read_all_docs_without_catterms(finTrainFile);  
	
	
	ofstream foutBestR;
	//foutBestR.open("bestRforcategs.txt");

	//makePredictionForCateg(5e-5, 30, 30);
	bestRforCategs(/*foutBestR*/);
	
	//finFreq.open("frequency(docs)");
	
	/* ofstream foutFreq;
	foutFreq.open("frequencyLocal.txt");
	getFrequencyLocal(foutFreq);
	
	ofstream foutFreq2;
	foutFreq2.open("frequencyGlobal.txt");
	getFrequency(foutFreq2);
	*/
	//ifstream finFreq;
	//finFreq.open("frequencyGlobal.txt");
	
	
	//knnBenchmark(finFreq);
	
	//processEntropyFile();	
	//testParameters();
	//ofstream foutMistakes;
	//foutMistakes.open("mistakesFind.txt");
	//findMistakes(20, foutMistakes);
	
	//readFreqFile(finFreq);
	
	/*int doc = 2127362;
	int categ = 29745;
	int chosendocs = 30; // как делала при составлении файла
	ofstream foutCheck;
	foutCheck.open("CheckCorrectEntropy.txt");
	checkEntropia(doc, categ, chosendocs, foutCheck);
	*/
	
	

	return 0;
}