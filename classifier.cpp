#include "csvstream.hpp"
#include <iostream>
#include <set>
#include <cmath>

using namespace std;

class Classifier{
    public:

    // EFFECTS: Return a set of unique whitespace delimited words
    set<string> unique_words(const string &str) {
     istringstream source(str);
     set<string> words;
     string word;
     while (source >> word) {
      words.insert(word);
    }
    return words;
  }

    void getstats(istream& fin)
    {
        csvstream cvsin(fin, ',', true); // I don't know whether strict should be false
        vector<string> header = cvsin.getheader();
        int colnum_tag, colnum_content;
        for(int n = 0; n < header.size(); n++)
        {
            if(header[n] == "tag") colnum_tag = n;
            if(header[n] == "content") colnum_content = n;
        }


        //sift through file and find tags + content
        while (true) {
            vector<pair<string, string>> row; // to extract row from file in order to sift
            if (!(cvsin >> row)) break;
        
            data.push_back(make_pair(row[colnum_content].second, row[colnum_tag].second));
        }
        total_posts = data.size();

        //find the unique labels including student and teacher
        for (const auto &post : data) {
            labels[post.second]++;
        }
        labels_size = labels.size();
        
        //find the number of unique words (vocab) and how many times the word shows up
        for (const auto &post : data) {
            set<string> words = unique_words(post.first);
            for (const string &word : words) {
                vocab[word]++;
            }
        }

        //find number of times each word and label appear together
        for(const auto &post : data)
        {
            const string &content = post.first;
            const string &label = post.second;
            set<string> words = unique_words(content);

            for (const string &word : words) {
            pair_count[label][word]++;
            }
        }
        vocab_size = vocab.size();

    }


    double log_probability_score(const string &label, const string &content)
    {

        set<string> words = unique_words(content);


        //log-prior
        double log_prior = log(static_cast<double>(labels[label]) / total_posts);
        //log-likelihood
        double log_like = 0;
        for(const auto &word : words)
        {
            if(!pair_count[label][word] && vocab[word]) // word doesn't exist with label
            log_like += log(static_cast<double>(vocab[word])/total_posts);
            else if(!vocab[word]) //word isn't in vocab
            log_like += log(1.0/total_posts);
            else  //normal case
            log_like += log(static_cast<double>(pair_count[label][word])/labels[label]);
        }
        return log_prior + log_like;
    }

    string predict(const string &content)
    {
        string ret_label;
        double max_prob = -INFINITY;
        for(const auto &[label, freq] : labels)
        {
            double score = log_probability_score(label, content);
            if(score > max_prob || 
            (score == max_prob && (ret_label.empty() || label < ret_label)))
            {
                max_prob = score;
                ret_label = label;
            }

        }
        return ret_label;
    }

    void print(bool train_only)
    {
        //print training data
        if(train_only)
        {
            cout << "training data:" << endl;
            for(const auto &post : data)
            cout << "  label = " << post.second << ", content = " << post.first << endl;
        }
            cout << "trained on " << total_posts << " examples" << endl;
            if(train_only)
        {
            cout << "vocabulary size = " << vocab_size << endl << endl;
            cout << "classes:" << endl;
            for(const auto &[label, freq] : labels)
            {
            cout << "  " << label << ", " << freq << " examples, log-prior = " 
            << log(static_cast<double>(labels[label]) / total_posts) << endl;
            }
            cout << "classifier parameters:" << endl;
            for(const auto &[label, word_count] : pair_count)
            {
                for(const auto &[word, freq] : word_count)
                {
                    cout << "  " << label << ":" << word << ", count = " << freq 
                    << ", log-likelihood = " 
                    << log(static_cast<double>(pair_count[label][word])/labels[label]) 
                    << endl;
                }
            }
        }
        cout << endl;
    }

    void print_predict(ifstream &test_file)
    {
        csvstream test_stream(test_file, ',', true);
        vector<string> header = test_stream.getheader();
        int colnum_tag = -1, colnum_content = -1;
    
        for (int i = 0; i < header.size(); ++i) {
            if (header[i] == "tag") colnum_tag = i;
            if (header[i] == "content") colnum_content = i;
        }
    
        vector<pair<string, string>> row;
        int correct = 0, total = 0;
    
        cout << "test data:" << endl;
    
        while (test_stream >> row) {
            string actual_label = row[colnum_tag].second;
            string content = row[colnum_content].second;
            string predicted_label = predict(content);
            double log_prob = log_probability_score(predicted_label, content);
    
            cout << "  correct = " << actual_label
                 << ", predicted = " << predicted_label
                 << ", log-probability score = " << log_prob << endl;
            cout << "  content = " << content << endl << endl;
    
            if (predicted_label == actual_label) {
                correct++;
            }
            total++;
        }
    
        cout << "performance: " << correct << " / " << total 
        << " posts predicted correctly" << endl;
    }
    
    private:
    int total_posts,labels_size, vocab_size;
    vector<pair<string, string>> data; // <content, tag> 
    map<string, int> labels; //unique tags including student and teacher with frequency
    map<string, int> vocab; 
    // unique words in post and how many times word appears in data
    map<string, map<string, int>> pair_count; 
    //count how many times each word and label appear together

};

int main(int argc, char **argv){
    cout.precision(3);
    if(argc != 2 && argc != 3) 
    {
        cout << "Usage: classifier.exe TRAIN_FILE [TEST_FILE]" << endl;
        return -1;
    }
    string train_file_name = argv[1];
    ifstream train_file(train_file_name);
    if(!train_file)
    {
        cout << "Error opening file: " << train_file_name << endl;
        return -1;
    }
    bool train_only = true;
    ifstream test_file;
    if(argc == 3) 
    {
        train_only = false;
        string test_file_name = argv[2];
        test_file.open(test_file_name);
        if(!test_file)
        {
            cout << "Error opening file: " << test_file_name << endl;
            return -1;   
        }
    }

    Classifier clf;
    clf.getstats(train_file);

    clf.print(train_only);
    if (!train_only)
    clf.print_predict(test_file);




    return 0;
}