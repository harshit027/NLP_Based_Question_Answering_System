import sys
import nltk
# nltk.download('all') -- UNCOMMENT THIS
from nltk.corpus import stopwords
from nltk.tokenize import *
from Story import *
from Question import *
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tree import Tree
from time import gmtime, strftime

directoryName = ''
storyIds = []  # All the story ids in the input file
storyObjects = []
answerFile = None


def readInputFile():
    inputFile = open("testset1-inputfile.txt", 'r')  # Read from command line
    #inputFile = open("inputAll.txt", 'r')  # Read from command line
    global directoryName
    directoryName = inputFile.readline().rstrip("\n")
    for line in inputFile:
        storyIds.append(line.rstrip("\n"))


def readQuestionsFile(storyId):
    questionFile = open(directoryName.replace("\\", "/") + "/" + storyId + ".questions", "r")
    readQuestions = []
    while True:
        ques = Question()
        # Question Id of the question
        questionId = questionFile.readline()
        if not questionId: break
        colonIndex = questionId.index(":")
        ques.quesId = questionId[colonIndex + 2:]
        # print(ques.quesId)
        # Question of the question
        question = questionFile.readline()
        colonIndex = question.index(":")
        ques.ques = question[colonIndex + 2:]
        # print(ques.ques)
        # Question Type of the question
        questionText = question[colonIndex + 2:]
        questionText = questionText.replace(",", "")
        questionText = questionText.replace(".", "")
        spaceIndex = questionText.index(" ")
        firstWord = questionText[0:spaceIndex]
        quesTypes = ['why', 'what', 'where', 'who', 'when', 'how']
        if firstWord.lower() in quesTypes:
            ques.quesType = firstWord.lower()
        else:
            for qtype in quesTypes:
                if qtype in ques.ques.lower():
                    ques.quesType = qtype
        # Question Difficulty of the question
        questionDiff = questionFile.readline()
        colonIndex = questionDiff.index(":")
        ques.difficulty = questionDiff[colonIndex + 2:]
        # print(ques.difficulty)
        questionFile.readline()
        readQuestions.append(ques)
    return readQuestions


def readStoryFiles():
    for storyId in storyIds:
        storyFile = open(directoryName.replace("\\", "/") + "/" + storyId + ".story", 'r')
        story = Story()
        # Headline of the story
        headline = storyFile.readline()
        colonIndex = headline.index(":")
        story.headline = headline[colonIndex + 2:]
        # Date of the story
        date = storyFile.readline()
        colonIndex = date.index(":")
        story.date = date[colonIndex + 2:]
        # Story ID of the story
        id = storyFile.readline()
        story.storyId = storyId
        # Sentences of the story
        storyFile.readline()
        storyFile.readline()
        fileContent = storyFile.read()
        fileContent = fileContent.replace("\\", "")
        fileContent = fileContent.replace("\'s", "")
        fileContent = fileContent.replace("\'", "")
        sent_detector = load('tokenizers/punkt/english.pickle')
        story.sentences = sent_detector.tokenize(fileContent.strip())
        # Questions of the story
        story.questions = readQuestionsFile(storyId)
        sentenceProcessing(story)
        storyObjects.append(story)


def sentenceProcessing(story):
    lemmatizer = WordNetLemmatizer()
    stopWords = stopwords.words('english')
    for sentence in story.sentences:
        originalSentence = sentence
        sentence = sentence.replace(".", "")
        sentence = sentence.replace(",", "")
        sentence = sentence.replace("\n", " ")
        sentence = sentence.replace("\s", "")
        sentence = sentence.replace("\\", "")
        sentenceWords = word_tokenize(sentence)
        story.sentWords[sentence] = sentenceWords
        filteredWords = []
        for word in sentenceWords:
            if word not in stopWords:
                filteredWords.append(word)

        filteredWords_Lemmatized = [lemmatizer.lemmatize(word) for word in filteredWords]
        story.sentLemmaWords[sentence] = filteredWords_Lemmatized
        story.allWordsPosTags[sentence] = nltk.pos_tag(sentenceWords)
        story.sentPosTags[sentence] = nltk.pos_tag(filteredWords_Lemmatized)


def removeCommonWords(sentence, question):
    newSentence = ""
    question = [word.lower() for word in question]

    for word in sentence:
        if word.lower() not in question:
            newSentence = newSentence + word + " "

    return newSentence


def wordMatch(story, ques):
    quesWords = word_tokenize(ques.ques)
    lemmatizer = WordNetLemmatizer()
    quesWords_Lemmatized = [lemmatizer.lemmatize(word) for word in quesWords]
    maxSentence = ''
    maxScore = 0
    maxSentenceWords = []
    for sent in story.sentences:
        sent = sent.replace(".", "")
        sent = sent.replace(",", "")
        sent = sent.replace("\n", " ")
        sent = sent.replace("\s", "")
        sent = sent.replace("\\", "")
        score = 0
        filteredWords_Lemmatized = story.sentLemmaWords[sent]
        postags = story.sentPosTags[sent]

        dict = {}
        propernouns = []
        referencetohuman = 'false'
        for tag in postags:
            dict[tag[0]] = tag[1]
            if 'NNP' in tag[1]:
                propernouns.append(tag[0])
            if 'NN' in tag[1]:
                referencetohuman = 'true'

        # Rule 1
        for qWord in quesWords_Lemmatized:
            if qWord in filteredWords_Lemmatized:
                if 'VB' in dict[qWord]:
                    score += 6
                    # break
                else:
                    score += 3
                    # break

        # Rule 2
        for pn in propernouns:
            if pn in quesWords_Lemmatized:  # The same noun word is present in the ques as well
                score += 6

        # Rule 3
        quesposttags = nltk.pos_tag(quesWords_Lemmatized)
        for tag in quesposttags:
            if 'NNP' in tag[1] and 'name' in filteredWords_Lemmatized:
                score += 4

        # Rule 4
        if propernouns.__len__() > 0 or referencetohuman == 'true':
            score += 4

        if score >= maxScore:
            maxScore = score
            maxSentence = sent
            maxSentenceWords = story.sentWords[sent]

    print("Answer: " + removeCommonWords(maxSentenceWords, quesWords))
    finalString = "\nAnswer: " + removeCommonWords(maxSentenceWords, quesWords) + "\n\n"
    answerFile.write(finalString)


def whoQuestions(story, ques):
    # Rule 1
    wordMatch(story, ques)


def whenQuestions(story, ques):
    timeKeywords = ['first', 'last', 'since', 'ago']
    sentKeywords = ['start', 'begin', 'since', 'year']
    quesKeywords = ['start', 'begin']
    TIME = ['morning', 'evening', 'tomorrow', 'soon', 'yesterday', 'last', 'week', 'recently', 'hour', 'ago', 'while',
            'past', 'present', 'january', 'february', 'march',
            'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'sunday',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday',
            'noon', 'month', 'year', 'everyday', 'day', 'today', 'seconds', 'minute', 'night', 'time', 'sunrise',
            'sunset', 'decade']
    for year in range(1400, 2016):
        TIME.append(year)

    quesWords = word_tokenize(ques.ques)
    lemmatizer = WordNetLemmatizer()
    quesWords_Lemmatized = [lemmatizer.lemmatize(word) for word in quesWords]

    maxScore = 0
    maxSentence = ''
    maxSentenceWords = []
    for sent in story.sentences:
        sent = sent.replace(".", "")
        sent = sent.replace(",", "")
        sent = sent.replace("\n", " ")
        sent = sent.replace("\s", "")
        sent = sent.replace("\\", "")
        score = 0
        # Rule 1
        sentenceWords = story.sentWords[sent]
        tree = nltk.ne_chunk(story.allWordsPosTags[sent])

        runWordMatch = False
        for i in tree:
            if type(i) == Tree and 'DATE' in i.label():
                score += 4
                runWordMatch = True
                break

        if not runWordMatch:
            for word in sentenceWords:
                if word.lower() in TIME:
                    score += 4
                    runWordMatch = True

        if runWordMatch:
            # WordMatch
            filteredWords_Lemmatized = story.sentLemmaWords[sent]

            postags = story.sentPosTags[sent]
            dict = {}
            propernouns = []
            referencetohuman = 'false'
            for tag in postags:
                dict[tag[0]] = tag[1]
                if 'NNP' in tag[1]:
                    propernouns.append(tag[0])
                if 'NN' in tag[1]:
                    referencetohuman = 'true'

            for qWord in quesWords_Lemmatized:
                if qWord in filteredWords_Lemmatized:
                    if 'VB' in dict[qWord]:
                        score += 6
                        # break
                    else:
                        score += 3
                        # break

        # Rule 2
        if 'the' in quesWords_Lemmatized and 'last' in quesWords_Lemmatized and any(key.lower() in story.sentLemmaWords[sent] for key in timeKeywords):
            score += 20

        # Rule 3
        if any(key.lower() in quesWords_Lemmatized for key in quesKeywords) and any(key1.lower() in story.sentLemmaWords[sent] for key1 in sentKeywords):
            score += 20

        if score >= maxScore:
            maxScore = score
            maxSentence = sent
            maxSentenceWords = story.sentWords[sent]

    print("Answer: " + removeCommonWords(maxSentenceWords, quesWords))
    finalString = "\nAnswer: " + removeCommonWords(maxSentenceWords, quesWords) + "\n\n"
    answerFile.write(finalString)


def getBestSents(story, ques):
    bestSentences = []
    scoreSentenceDict = {}
    quesWords = word_tokenize(ques.ques)
    lemmatizer = WordNetLemmatizer()
    quesWords_Lemmatized = [lemmatizer.lemmatize(word) for word in quesWords]
    maxScore = 0
    for sent in story.sentences:
        origSent = sent
        sent = sent.replace(".", "")
        sent = sent.replace(",", "")
        sent = sent.replace("\n", " ")
        sent = sent.replace("\s", "")
        sent = sent.replace("\\", "")
        score = 0
        filteredWords_Lemmatized = story.sentLemmaWords[sent]
        postags = story.sentPosTags[sent]
        dict = {}
        propernouns = []
        for tag in postags:
            dict[tag[0]] = tag[1]
            if 'NNP' in tag[1]:
                propernouns.append(tag[0])
            if 'NN' in tag[1]:
                referencetohuman = 'true'

        # Rule 1
        for qWord in quesWords_Lemmatized:
            if qWord in filteredWords_Lemmatized:
                if 'VB' in dict[qWord]:
                    score += 6
                    # break
                else:
                    score += 3
                    # break

        scoreSentenceDict[sent] = score
        if score > 0:
            bestSentences.append(origSent)
        if score > maxScore:
            maxScore = score

    finalSent = []
    for sent in scoreSentenceDict:
        if scoreSentenceDict[sent] == maxScore:
            finalSent.append(sent)
    bSentences = sorted(scoreSentenceDict, key=scoreSentenceDict.get, reverse=True)
    return bestSentences


def whyQuestions(story, ques):
    quesWords = word_tokenize(ques.ques)
    bestSentences = getBestSents(story, ques)
    precedingSentences = []
    followingSentences = []
    for i in range(0, story.sentences.__len__()):
        if story.sentences[i] in bestSentences:
            if i + 1 < story.sentences.__len__() and not followingSentences.__contains__(story.sentences[i + 1]):
                followingSentences.append(story.sentences[i + 1])
            if i - 1 >= 0 and not precedingSentences.__contains__((story.sentences[i-1])):
                precedingSentences.append(story.sentences[i - 1])

    maxSentence = ''
    maxScore = 0
    maxSentenceWords = []
    # Rule 1
    for sent in story.sentences:
        origSent = sent
        sent = sent.replace(".", "")
        sent = sent.replace(",", "")
        sent = sent.replace("\n", " ")
        sent = sent.replace("\s", "")
        sent = sent.replace("\\", "")
        score = 0
        if origSent in bestSentences:
            score += 3
        if origSent in precedingSentences:
            score += 3
        if origSent in followingSentences:
            score += 4
        if 'want' in story.sentWords[sent] or 'Want' in story.sentWords[sent]:
            score += 4
        if 'so' in story.sentWords[sent] or 'because' in story.sentWords[sent] or 'Because' in story.sentWords[sent] or 'So' in story.sentWords[sent]:
            score += 4

        if score >= maxScore:
            maxScore = score
            maxSentence = sent
            maxSentenceWords = story.sentWords[sent]

    print("Answer: " + removeCommonWords(maxSentenceWords, quesWords))
    finalString = "\nAnswer: " + removeCommonWords(maxSentenceWords, quesWords) + "\n\n"
    answerFile.write(finalString)


def whereQuestions(story, ques):
    quesWords = word_tokenize(ques.ques)
    lemmatizer = WordNetLemmatizer()
    quesWords_Lemmatized = [lemmatizer.lemmatize(word) for word in quesWords]
    locationPreps = ['in', 'on', 'at', 'by', 'near', 'nearby', 'above', 'below', 'over', 'under', 'up', 'down',
                     'around', 'through', 'inside', 'outside', 'between', 'beside', 'from', 'opposite',
                     'beyond', 'front', 'back', 'behind', 'next', 'top', 'within', 'beneath',
                     'underneath', 'among', 'along', 'against']

    maxSentence = ''
    maxScore = 0
    maxSentenceWords = []
    for sent in story.sentences:
        sent = sent.replace(".", "")
        sent = sent.replace(",", "")
        sent = sent.replace("\n", " ")
        sent = sent.replace("\s", "")
        sent = sent.replace("\\", "")
        score = 0
        sentenceWords = story.sentWords[sent]
        filteredWords_Lemmatized = story.sentLemmaWords[sent]

        postags = story.sentPosTags[sent]
        dict = {}
        propernouns = []
        referencetohuman = 'false'
        for tag in postags:
            dict[tag[0]] = tag[1]
            if 'NNP' in tag[1]:
                propernouns.append(tag[0])
            if 'NN' in tag[1]:
                referencetohuman = 'true'

        # Rule 1
        for qWord in quesWords_Lemmatized:
            if qWord in filteredWords_Lemmatized:
                if 'VB' in dict[qWord]:
                    score += 6
                else:
                    score += 3

        # Rule 2
        for word in sentenceWords:
            if word in locationPreps:
                score += 4
                break

        # Rule 3
        sentTags = story.allWordsPosTags[sent]
        tree = nltk.ne_chunk(sentTags)
        for i in tree:
            if type(i) == Tree and 'LOCATION' in i.label():
                score += 6

        if score >= maxScore:
            maxScore = score
            maxSentence = sent
            maxSentenceWords = story.sentWords[sent]

    print("Answer: " + removeCommonWords(maxSentenceWords, quesWords))
    finalString = "\nAnswer: " + removeCommonWords(maxSentenceWords, quesWords) + "\n\n"
    answerFile.write(finalString)


def whatQuestions(story, ques):
    months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
              "november", "december"]
    quesWords = word_tokenize(ques.ques)
    lemmatizer = WordNetLemmatizer()
    quesWords_Lemmatized = [lemmatizer.lemmatize(word) for word in quesWords]
    maxSentence = ''
    maxScore = 0
    maxSentenceWords = []
    for sent in story.sentences:
        sent = sent.replace(".", "")
        sent = sent.replace(",", "")
        sent = sent.replace("\n", " ")
        sent = sent.replace("\s", "")
        sent = sent.replace("\\", "")
        score = 0
        sentenceWords = story.sentWords[sent]
        filteredWords_Lemmatized = story.sentLemmaWords[sent]

        postags = story.sentPosTags[sent]
        dict = {}
        propernouns = []
        referencetohuman = 'false'
        for tag in postags:
            dict[tag[0]] = tag[1]
            if 'NNP' in tag[1]:
                propernouns.append(tag[0])
            if 'NN' in tag[1]:
                referencetohuman = 'true'

        # Rule 1
        for qWord in quesWords_Lemmatized:
            if qWord in filteredWords_Lemmatized:
                if 'VB' in dict[qWord]:
                    score += 6
                    # break
                else:
                    score += 3
                    # break

        # Rule 2
        for word in quesWords:
            if word.lower() in months:
                for keyword in ["yesterday", "today", "tomorrow", "last", "night"]:
                    if keyword in sentenceWords:
                        score += 3
                        break
                break

        # Rule 3
        if "kind" in quesWords and ("call" in sentenceWords or "from" in sentenceWords):
            score += 4

        # Rule 4
        if "name" in quesWords:
            for word in ["name", "call", "known"]:
                if word in sentenceWords:
                    score += 20
                    break

        if score >= maxScore:
            maxScore = score
            maxSentence = sent
            maxSentenceWords = story.sentWords[sent]

    print("Answer: " + removeCommonWords(maxSentenceWords, quesWords))
    finalString = "\nAnswer: " + removeCommonWords(maxSentenceWords, quesWords) + "\n\n"
    answerFile.write(finalString)


def defaultQues(story, ques):
    quesWords = word_tokenize(ques.ques)
    lemmatizer = WordNetLemmatizer()
    quesWords_Lemmatized = [lemmatizer.lemmatize(word) for word in quesWords]

    maxSentence = ''
    maxScore = 0
    maxSentenceWords = []
    for sent in story.sentences:
        sent = sent.replace(".", "")
        sent = sent.replace(",", "")
        sent = sent.replace("\n", " ")
        sent = sent.replace("\s", "")
        sent = sent.replace("\\", "")
        score = 0
        filteredWords_Lemmatized = story.sentLemmaWords[sent]
        postags = story.sentPosTags[sent]

        dict = {}
        propernouns = []
        referencetohuman = 'false'
        for tag in postags:
            dict[tag[0]] = tag[1]
            if 'NNP' in tag[1]:
                propernouns.append(tag[0])
            if 'NN' in tag[1]:
                referencetohuman = 'true'

        # Rule 1
        for qWord in quesWords_Lemmatized:
            if qWord in filteredWords_Lemmatized:
                if 'VB' in dict[qWord]:
                    score += 6
                    # break
                else:
                    score += 3
                    # break

        if score >= maxScore:
            maxScore = score
            maxSentence = sent
            maxSentenceWords = story.sentWords[sent]

    print("Answer: " + removeCommonWords(maxSentenceWords, quesWords))
    finalString = "\nAnswer: " + removeCommonWords(maxSentenceWords, quesWords) + "\n\n"
    answerFile.write(finalString)

def beginAnswering():
    global answerFile
    answerFile = open("answer.response", "w")
    for story in storyObjects:
        for ques in story.questions:
            print("Question: " + ques.quesId.replace("\n", ""))
            answerFile.write("QuestionID: " + ques.quesId.replace("\n", ""))
            if ques.quesType == "who":
                whoQuestions(story, ques)
            elif ques.quesType == "where":
                whereQuestions(story, ques)
            elif ques.quesType == "why":
                whyQuestions(story, ques)
            elif ques.quesType == "when":
                whenQuestions(story, ques)
            elif ques.quesType == "what":
                whatQuestions(story, ques)
            else:
                #whoQuestions(story, ques)
                defaultQues(story, ques)


if __name__ == '__main__':
    print("Started at: " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) )
    readInputFile()
    print("Input file read.")
    print("Processing the story files in the input file")
    readStoryFiles()
    print("The story files processed.")
    beginAnswering()
