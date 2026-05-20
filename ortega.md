```
Knowledge-Based Systems 235 (2022) 107597
```
```
Contents lists available at ScienceDirect
```
# Knowledge-BasedSystems

```
journal homepage: http://www.elsevier.com/locate/knosys
```
# Multi-viewinformedattention-basedmodelforIronyandSatire

# detectioninSpanishvariants

## ReynierOrtega-Buenoa,

### ∗

## ,PaoloRossoa,JoséE.MedinaPagolab

a _PRHLT Research Center, Universitat Politècnica de València, Valencia, Spain_
b _Universidad de Ciencias Informáticas, Havana, Cuba_

### a r t i c l e i n f o

_Article history:_
Received 25 January 2021
Received in revised form 5 October 2021
Accepted 12 October 2021
Available online 22 October 2021

_Keywords:_
Ironyandsatire
Attentionmechanism
Linguisticfeatures
Contextualizedpre-trainedembedding
Fusingrepresentation
Spanishvariants
Figurativelanguage

### a b s t r a c t

```
Making machines understand language and reasoning on it has been one of the most challenging
problems addressed by Artificial Intelligent researchers. This challenge increases when figurative
language is used for communicating complex meanings, intentions, emotions and attitudes in creative
and funny ways. In fact, sentiment analysis approaches struggle when facing irony, satire and other
figurative languages, particularly those where the explanation of a prediction might arguably be as
necessaryasthepredictionitself.ThispaperdescribesanewmodelMvAttLSTMbasedondeeplearning
for irony and satire detection in tweets written in distinct Spanish variants. The proposed model is
based on an attentive-LSTM informed with three additional views learned from distinct perspectives.
WeinvestigatetwostrategiestopasstheseviewsintoMvAttLSTM.Weperformanextensiveevaluation
on three corpora, one for irony detection and two for satire detection. Moreover, in order to study the
robustness of our proposed model, we investigate its performance on humor recognition. Experiments
confirm that the proposed views help our model to improve its performance. Moreover, they show
that affective information benefits our model to detect irony and satire. In particular, a first analysis of
the results highlights the discriminating power of emotional features obtained from SenticNet and SEL
lexicon. Overall, our system achieves the state-of-the-art performance in irony and satire detection in
Spanish variants and competitive results in humor recognition.
©2021ElsevierB.V.Allrightsreserved.
```
**1. Introduction**

Languageitselfisaperfectillustrationofhumancreativity,and
it achieves its splendor when some semantics rules and maxims
ofhumancommunication[1,2]aredisrespectedtocreateexpres-
sions whose real meaning diverges from what it is apparently
said.Thispeculiarusageoflanguagewithcreativeandfunnypur-
poses has been coined with the term _Figurative Language_ [3–5].
Irony, satire, sarcasm, humor, puns, simile, hyperbole, metonym
andmetaphorareforms(ordevices)offigurativelanguage.While
it is true that all these forms are used to communicate complex
meanings, not all of them are used by common people. Some
forms are relegated only to literary and poetry usages [6].
Irony and satire are pervasive and popular in everyday com-
munication. As human beings, we appeal to these devices as
effective ways through which literal meanings are intentionally
deviated in favor of secondary interpretations. Particularly, both
devicesareacknowledgedtoexpressanattitudethatisgenerally
negativeandimplicitbehindanapparentpositivemessage.Thus,
they are frequently used to criticize, complain, ridicule or mock.

```
∗Corresponding author.
E-mail address: rortega@prhlt.upv.es (R. Ortega-Bueno).
```
```
Even when there are commonalities between both phenomena,
irony seem to be more primitive and universal than satire.
The most common types of irony used in social media are
situational and verbal irony. On the one hand, situational irony
referstospecificeventsthatfailtomeetexpectations[7]e.g. ‘‘The
fire station burns down while the firemen are out on a call’’. On
the other hand, verbal irony has been traditionally identified as
figurativedevicewhereenunciatedwordsimplysomethingother
than their literal meanings. In other words, their real meaning is
opposite to the literal one and it needs to be inferred through
interpretation e.g. ‘‘A burned tongue is a lovely way to start the
day’’. Sarcasm is often considered a specific type of verbal irony
which has a more aggressive tone [8], is directed toward an
individual or a group [9–12], and is used intentionally [13,14].
An example of sarcasm would be the exclamation ‘‘You’re really
brilliant!’’ about someone who has done a foolish act.
Satire is an interesting concept which is strongly related with
ironyandhumor.Ittakesadvantageofindirectspeechandnega-
tiveattitudeimplicitinirony.Thisdevicealsoappealstofeatures
of humor such as: parody, exaggeration, juxtaposition, compari-
son,analogy,anddoubleentendreswithcensoringpurpose.Satir-
ical messages may be aggressive and offensive, but they always
have a deeper meaning and a social signification beyond that of
```
https://doi.org/10.1016/j.knosys.2021.
0950-7051/©2021 Elsevier B.V. All rights reserved.


the humor [15]. Satire does not make sense when the reader
does not understand the real intent hidden in the ironic/funny
dimension; like in irony, the real meaning of a satirical message
lays in the figurative interpretation of the content. Satire can
be separated in two distinct directions: _Juvenalian_ or _Horatian_
styles[16].Ontheonehand,the _Juvenalian_ styleofsatireisbased
on ridicule and sarcasm. On the other hand, the _Horatian_ style
contains tease and humor.
Irony and satire have been studied from many disciplines
such as Linguistics, Psychology, Rhetoric, Pragmatics, Semantics,
etc., however, they are not only enclosed to these theoretical
studies.Nowadays,bothdevicesaretypicallyusedinsocialmedia
platformstofavorsocialinteractions,evokinghumor[17],dimin-
ishing or enhancing criticism [18,19], and getting the attention
of the readers by means of the creativity [20]. These forms of
figurative language have great impact on several other Natural
Language Processing (NLP) tasks that aiming at monitoring social
media content. In some cases the presence of ironic message
plays a specific role: _‘‘implicit polarity reversal’’_. This means, that
a message seems to be positive but its real meaning is negative
(or vice-versa). Due to this peculiarity of ironic and sarcastic
expressions, the sentiment analysis approaches decline when
facing irony in social media texts [21–26]. In fact, this prob-
lem become more challenging in sentiment analysis approaches
where the explanation of the results is more important than the
decision itself [27,28]. Sarcasm as a specific sub-type of verbal
irony has implications to cope with the bad phenomenon of mis-
communication, particularly: hate, aggressiveness, and nastiness
speech[29].Ignoringthepresenceofsarcasmcausesthattheim-
plicitmeaning,generallyhurtfulandoffensive,bemisunderstood
with the results of exposing people to toxic information. In this
context also it results crucial to understand what pieces of mes-
sage are relevant. Recently, interesting evidence about the use of
satire to disguise fake news has been discussed in [30,31]. For
people, understanding satire as fake messages may deprive them
of desirable entertainment content, while recognizing fake infor-
mation as legitimate satire may expose them to disinformation
or misinformation.
Following the timeline of computational methods for irony
andsatiredetection,itispossibletoenvisagetwodistinguishable
approachesnamely,classicalfeatures-basedmachinelearningap-
proach,anddeep-leaningapproach.Intheliteraturemanyworks
explored several linguistic, stylistic, content, affective, and con-
textual features to address the problem in a shallow supervised
way[32].Thehandcraftfeaturesderivedfromthisapproachhave
proved to be feasible when small dataset are provided.
In the last five years, deep learning techniques became very
popular in NLP, and applied to irony and satire detection. These
methods show a better performance than classical feature-based
machinelearningmodels.Inthisscenario,avastnumberofworks
are based on attentive-Recurrent Neural Networks (att-RNN),
particularly by using of Long Short Term Memory (LSTM) [33]
andGatedRecurrentUnit(GRU)[34]withattentionmechanisms,
which have been effective to capture complex dependencies
among words within the text and pay more attention to those
wordsthatincreasetheeffectivenessofthesenetworksinseveral
tasks of NLP [35–38].
Recently, a groundbreaking advance in NLP has been marked
by using transformer-based network architectures [39] which
opened a new avenue for training robust and contextual-aware
word embeddings in unsupervised manner. The use of these
pre-trained contextualized models has been widely spread by
means of Bidirectional Encoder Representations from Transform-
ers (BERT) [40] and BERT’s family architectures [41–44]. Clearly,
this spreading has reached figurative language processing, and
cause that these new computational methods outperformed the
state of the art by a substantial margin [45–47].

```
The nested non-linear functions of deep learning algorithms
provokethesemodelsareusuallyappliedinablack-boxmanner,
that is, no interpretable knowledge is provided about what ex-
actly causes them to arrive at their predictions. In this sense, the
attention mechanisms became a (albeit narrow) way for dealing
withtheproblemofmodelinterpretability.Recently,attentionas
a way of explainable deep learning method became a very pop-
ular and controversial topic. Some works claimed that attention
weightsdonotprovidemeaningful‘‘explanations’’forsupporting
the final predictions [48,49], however other works state that
they are able for discovering how neural models capture several
linguistic notions of syntax, semantic and coreference [50–52].
Despitediverseviewsonthematter,empiricalresultsonthetask
of binary irony classification show that attention mechanisms
are able for capturing ironic cues, word polarity and explicit and
implicit sentiment incongruity [45,53].
Evenwhen,remarkableadvanceshavebeenobservedinirony
andsatiredetection,theseadvanceshaveshowedanasymmetric
developmentwithrespecttolanguagesotherthanEnglish.Thisis
the case of Spanish and its variants, where few researchers have
addressed the problem. In the case of satire, only the works [54,
55]havestudiedthephenomenonintheSpanishlanguagebyus-
ing a machine learning-based approach. Irony detection in Span-
ishvariantshasbeensurveyedin[56],butfewmethodsreliedon
deep learning approaches [57–60]. According to our knowledge
only the works [45,60] take advantages of contextualized pre-
trained word embedding. For that, more efforts must be paid
to study irony and satire in Spanish language. In this work we
propose an attentive-based deep learning method in order to
investigate further the detection of irony and satire in Spanish:
```
- Irony and satire are pragmatic phenomena, hence both are
    contextual-dependents. Additional knowledge such as: lan-
    guage and its variety, sociolinguistics and cultural back-
    ground are crucial for precisely recognize and understand
    these forms of figurative language.
- Many studies on irony and satire detection have been con-
    ducted from three directions: linguistics features with ma-
    chine learning approaches, deep learning techniques based
    on att-RNN, and recently by means of contextualized pre-
    trained word embeddings with a single-modality represen-
    tation of the texts. However, there are no works that pay
    attentiontoexploreironyandsatireinSpanishfromafusion
    information perspective where these three approaches are
    fused aiming to outperform the state of the art.

```
To overcome these challenges, we aim at addressing the fol-
lowing research questions:
```
```
RQ1. Could irony and satire detection methods take advantage
of combining multiple representations (views) of text in
terms of linguistic-based representation, universal sentence
encoder-based representation, and contextualized
pre-trained embeddings?
RQ2. How the proposed views can be effectively combined to
proper inform an attentive recurrent model?
RQ3. Are multiple heads of attention (multi-head) more feasi-
ble than single attention (self) for capturing multiples and
complexrelationsamongwordsinironicandsatiricaltexts?
```
```
With the aim to answer the formulated research questions,
```
### in this work we propose a new model ( M v AttLSTM ) which re-

```
lies on a multi-view informed attentive-LSTM neural network.
We consider an attentive recurrent model due to the attention
mechanismsallowthemodeltofocusandplacemore‘‘attention’’
on the relevant parts of the text sequence in order to capture
complex syntactic and semantic properties used in ironic and
satiricalmessages.Specifically,welearnthreeindependentviews
```

for each text, and we pass them to our MvAttLSTM. The first
one( _Linguistic-view_ )isbasedonseverallinguisticsfeatureswhich
have proved to be strong cues for discriminating both irony
and satire. The second one considers a deep dense encoding of
the text by means of Multilingual Universal Sentence Encoder
( _MUSE-view_ ). And, finally the last one ( _BERT-view_ ) considers a
contextualized pre-trained embedding obtained after a tuning of
the BERT model. We evaluate the effectiveness of our method
on one corpus for irony detection and on two distinct corpora
for satire detection. For irony detection, the corpus is the one
proposed for the _IroSvA’19_ shared task: _Irony detection in Spanish
Variants_ [56] was used, whereas, in the case of satire detection
task the corpora introduced in [54,55] were employed. Our pro-
posal outperformed both previous systems participating in the
_IroSvA’19_ shared task and recent methods [45,61]. Also, for satire
detectionourproposaloutperformedpreviousmethodsbyasub-
stantial margin in both corpora. Additionally, we provide several
analysesinordertoevaluatetwostrategiesforfusingthelearned
viewsintoMvAttLSTMandinvestigatetheimpactofeachviewon
the performance of MvAttLSTM. Finally, an interesting analysis
is carried out on the attention mechanism to observe how our
proposaltakesintoaccountsomefeaturesrelatedwithironyand
satire such affective content. In short, the major contributions of
this paper are summarized below:

- Toinvestigatetheproblemofcomputationalironyandsatire
    detection in Spanish variants in three widely used corpora.
    Moreover, taking into account the closed relation among
    irony, satire and humor, we evaluate the robustness of the
    proposed model on humor recognition.
- To propose a novel approach (MvAttLSTM) based on repre-
    sentation fusion. Particularly, efficient representations from
    three distinct perspectives are computed and combined to
    informanattentive-LSTMmodel.Theproposedmethodout-
    performs the state of the art approaches in satire and irony
    detection in Spanish and obtains competitive results in hu-
    mor recognition.
- To investigate distinct forms of combining the proposed
    views, also to evaluate the impact of each view on the
    proposed model MvAttLSTM.
- To study the impact of two kinds of attention mechanisms,
    self attention vs. multi-head attention, on the proposed
    model.

The rest of this paper is structured as follows. Section 2 in-
troduces the state of the art for both irony and satire detection,
with special interest in those approaches proposed for Spanish.
Section3formalizesourproposalbasedonamulti-viewattention
based model. Particularly, we describe each one of the repre-
sentation used and two distinct ways of fusing these views for
irony and satire detection. In Section 4 a detailed description of
thecorpora,resources,preprocessingandtheexperimentalsetup
is introduced. Also, an exhaustive evaluation of our proposal
and a comparison with other approaches is presented. Moreover,
considering the closed relation among irony, satire and humor,
we evaluate the robustness of our model to recognize humor.
Finally, we draw some conclusions and discuss future work.

**2. State of the art**

There is a considerable amount of literature on computa-
tionally irony and satire detection [see 62–65]. In general, ap-
proaches to deal with these forms of figurative language can
be classified into: features-based machine learning approach and
deep leaning-based approach. Initially, machine learning method
combinedwithfeature-basedrepresentations(lexical,contextual,
stylistic, affective, discursive, etc.) received the most attention.

```
But,recently,deeplearning-basedapproachesaregaininginterest
due to the capacity of these models to automatically learn fea-
ture representations that are omitted in hand-craft extraction or
simply have abstraction levels beyond of human bounds. In this
line,thenexttwosubsectionssurveytherelevantworksforirony
and satire detection. Also, considering the imbalanced number
of studies in other languages than English, a third subsection
discusses irony and satire in a multilingual setting, with special
focus on Spanish.
```
```
2.1. Machine learning based approaches
```
```
Irony. Computational irony detection has been addressed by
the NLP community from different perspectives. In preliminary
works, the role of textual-based features obtained from the text
(suchasn-grams,punctuationmarks,part-of-speechtags,among
others) has been widely explored for its detection [66–70]. Other
works drew attention to theoretical aspects of irony such as
incongruity and opposition. Based on these aspects, features de-
rivedfromsemanticambiguity,synonyms,antonymsandpolarity
contrast have been studied in [25,71–73]. Many theories seem to
agree that an implicit attitude is expressed when being ironic.
Aiming to capture the relation between irony and subjectivity
in language, several approaches have focused on affective infor-
mation for improving irony detection [74–78]. Verbal irony is
without doubt a pragmatic phenomenon, hence, contextual and
extra-linguistic information result crucial for its detection and
comprehension. In this sense, information regarding the context
surrounding a given text has been exploited in order to deter-
mine whether a text has an ironic or sarcastic intention [79–82].
Discovering new features with discriminative power and topic-
independency have been the most active directions of machine
learning approaches. Regarding machine learning algorithms, the
most used have been Random Forest (RF), Decision Trees (DT),
Naïve Bayes (NB) and Support Vector Machines (SVM). Recently,
in [83] the impact of the imbalanced distribution of classes in
irony and sarcasms detection has been studied from a machine
learning perspective.
Satire. Machine learning has been the most used approach for
satire detection [54,55,84–86]. In the seminal paper of Burfoot
and Baldwin [84], the problem of detecting satire was explored
with simple bag of words features (BoW) using two feature-
weighting methods: (i) binary feature weighting and (ii) bi-
normal separation (BNS) features scaling. Further, lexical (head-
lines, profanity, slang) and semantic features were added to
enrichtextrepresentation.Tocomputethesemanticfeaturethey
identify the named entities in a given document and query the
web for the conjunction of those entities. In this direction, [85]
proposedtoextenttheBNSfeaturesscalingmethodwiththe tf-idf
weighting schema to improve satire detection in news genre.
In [54] studied the problem of satire detection in tweets.
Linguistic differences between satirical and factual content were
explored by mean of frequency, ambiguity, synonyms, part of
speech (PoS) tags, sentiments, characters, and slang words as
features. Experiments showed that some linguistic features are
topic-independentandhenceusefulcluestoaddresstheproblem.
In a same fashion, a psycho-linguistics approach was introduced
in [55] to identify satirical tweets. A wide variety of psycho-
logical and linguistic features from Linguistic Inquiry and Word
Count lexicon (LIWC) [87] were evaluated. Results confirmed the
usefulness of emotional, social, and psychological dimension for
satire detection. In [30] were considered five predictive features:
absurdity, humor, grammar, negative affect, and punctuation,
and applied an SVM method. After, combining three out of five
features (absurdity, grammar, and punctuation), the authors ob-
servedthattheBNSfeaturescalingissuitableforsatiredetection
and the model obtains good results.
```

Sensibility of lexical, linguistic and n-gram based features
across three textual genres was reported in [88]. Specifically,
the impact of features associated on affective words, acts of the
speech, sensorial words, and shallow clues of figurative device
(alliteration,grammaticalinversion,hyperbole,onomatopoeiaand
imaginary) was evaluated. Results showed that n-grams and
features related with the act of the speech were good as genre-
independentandhencetheyresultedsuitableforsatiredetection
in multiples genres. In a similar fashion, an emotions and sen-
timents based representation was proposed in [89] for satire
detectioninnewswires,Amazonproductreviewsandanin-house
Twitter corpus. Experiments were performed using the SVM and
RF methods. Results concluded the usefulness of the proposed
features for satire detection. In [90] the impact of emotions on
recognizing satirical texts from other figurative forms (humor,
irony, sarcasm) and factual language was analyzed.
Recently, satire has received more attention due to the com-
monalities with the undesirable phenomenon of misinformation
in social media, and particularly with fake news spreading [31,
91,92]. In order to reduce the exposure to misinformation in
social media, publishers of fake news have begun to masquerade
as satire sites to avoid being demoted. For users, incorrectly
recognizing satire as fake news may deprive them of desirable
entertainment content, while identifying a fake news story as
legitimate satire may expose them to misinformation.

_2.2. Deep learning-based approaches_

_Irony._ Recently,manydeeplearning-basedapproachesforad-
dressing irony detection have been proposed. Word embeddings,
Convolutional Neural Networks (CNNs), att-RNNs, and
Transformers-basedmodelshavebeenexploitedforcapturingthe
presenceofironyinsocialmediacontent[45,46,93–103].These-
mantic and syntactic properties of pre-trained word embeddings
havebeenhighlightedinseveralstudies[96,97,104].Forinstance,
word embeddings have been explored to capture incongruity
in text with non-affective words [96]. In the study [104], irony
detection was addressed using a multi-faceted representation
which fuses psycho-linguistic features with word embedding
vectors that were obtained by using Doc2Vec [105]. In [97] the
generalizationcapabilitiesofanunsupervisedtopicmodeltrained
for irony detection showed a substantial increasing when the
word embedding information was incorporated.
Several methods exploited the advantages of CNN for dis-
covering local features that result useful for irony detection.
An interesting idea was proposed in [98], which introduced a
framework for learning irony features from a corpus using CNN.
Thisapproachinvestigatedwhetherfeaturesextractedusingpre-
trained sentiment CNN, emotion CNN and personality CNN mod-
els can improve the overall performance. In another direction,
the role of the content and contextual information for sarcasm
detection taking advantages of a multi-view model were pre-
sented in [99]. For that purpose, two CNN models were trained
to generate stylometric and personality embeddings for each
user’s comments. Later, both embedding were fused in a multi-
view setting using Canonical Correlation Analysis (CCA) [106].
A content-based sentence representation was extracted using
another CNN and appended with context vectors to obtain the
final decision. In another study [93], a model that combines
dense neural networks (DNNs) with time-convolution and LSTM
(CNN-LSTM)wasproposedfordetectingsarcasmintweets.These
existing studies use the convolutional network to automatically
derive deep features from texts for irony detection. Results of
thesedeeplearning-basedmodelsaregenerallybetterthanthose
obtained with classical feature based machine learning methods.
RNNs have been used for addressing irony detection due to
their abilities for capturing long and short dependencies among

```
wordswithintexts.In[94]studiedtheroleplayedbytheconver-
sational context in a sarcasm reply. Particularly, results proved
that LSTMs that can model both the context and the sarcastic
reply achieve better performance than LSTMs that read only the
reply. In another direction, many approaches studied the impact
of attentive-based representation with linguistics features [100,
107]. Experiments have concluded that considering hand-crafted
features help models to increase their effectiveness. From an-
other point of view, the model introduced in [53], proposed
strategies to improve irony detection by transferring knowledge
from sentiment resources. This work proposed three different
attentive-LSTM approaches that differ in the way of including
the sentiment resources, either injecting the sentiment directly
to the attention mechanisms or merging the output of different
networks specialized on sentiment analysis and irony detection.
In a similar fashion, in [108] a multi-task learning approach was
proposed to leverage the knowledge in sarcasm detection and
sentimentanalysistask.Experimentsshowedthatthesetwotasks
are correlated, and training a deep neural network that models
this correlation in a multi-task learning setting improves the
performanceofbothtasks.Moreover,in[109]amulti-tasklearn-
ing framework for multi-modal sarcasm, sentiment, and emotion
analysis was proposed. The authors take advantage of the sen-
timent and emotions of the speaker to predict sarcasm. In the
multi-task framework, sarcasm was considered as the main task,
whereas emotion and sentiment detection were used as sec-
ondary tasks. Results confirmed that the multi-task framework
achieves better performance for the primary task, i.e. sarcasm
detection,withthehelpofemotionandsentimentanalysistasks.
The use of transformer-based models [39] has changed the
way of modeling and working with textual data in an unprece-
dented way. In fact, these models have been widely spread by
means of BERT [40] and other BERT’s related architectures [41–
44]. Clearly, this spreading has reached very fast also FL process-
ing: new methods based on transformer models outperformed
thestateoftheartinironydetectionbyasubstantialmargin[45–
47]. In this line, in [46] the RoBERTa model [42] was used to
encode the sentences, that was further contextualized by means
of a Recurrent Convolutional Neural Network to address irony
and sarcasm detection. This model outperformed state of the art
on four benchmark datasets for irony and sarcasm detection in
English. In [45] a simplification of the BERT architecture was
proposed to contextualize pre-trained word embeddings. Specif-
ically, this work contextualized Word2Vec word embeddings,
trained with several millions of tweets both for English and
Spanish. This strategy, opposite to the use of pre-trained BERT,
aimed to train the proposed model from in-domain data using
the same powerful backbone architecture as BERT. This model
outperformspreviousmodelsforironydetectioninSpanishshort
texts.
Satire. Notwithstanding a vast amount of deep learning-based
methods have been proposed for irony detection, and the com-
monalities between irony and satire, few methods have ad-
dressed the problem of satire detection from a deep learning
perspective. Recent works in this direction have been presented
in [110–112]. In [110] a four-level hierarchical network with at-
tention mechanism was presented to differentiate satirical news
from true ones. Psycholinguistics, writing stylistic, structural and
readability-based features were included to the model at both
paragraph and document level. The evaluation suggested that
readability features supported the overall classification while
psycholinguisticfeatures,writingstylisticfeatures,andstructural
featuresarebeneficialatparagraphlevel.Theanalysisofindivid-
ual features reveals that satirical news tend to be emotional and
imaginative. Another idea was explored in [111] which proposed
to use CNN, LSTM, and GRU to detect satire at both sentence and
documentlevels.Theyconcludedthatfine-grainedsentence-level
analysis provides an in-depth insight into the phenomenon of
satire.
```

_2.3. Multilingual approaches_

Most of the works on irony and satire detection have investi-
gated the problem in English. Notwithstanding, there have been
some efforts to investigate it in other languages such as: Chi-
nese [113], Czech [70], Dutch [69], French [114,115], Italian [24,
116,117],Portuguese[66],Spanish[56,118,119],andArabic[120,
121]. Even when in closely related tasks like sentiment analysis
have emerged an increasing number of works addressing the
multilinguality issue [122–128], where few works explored this
in the context of irony and satire detection. Taking advantage of
thefindingachievedformultilingualsentimentanalysiswouldbe
aninterestingdirectiontoimprovesatireaironyinthisscenario.
From a multilingual perspective, the approaches for irony and
satire detection can be analyzed in two main directions: (i) mul-
tilingual setting, where the model is trained and evaluated sepa-
ratelyoneachlanguage,(ii)cross-lingualsetting,wherethemod-
els is trained in one or more language and evaluated on another
differentone.Multilingualsettinghasbeenthemostinvestigated.
Prior works where presented in [70] and [113] for Czech–English
and Chinese–English languages respectively. In [129] a novel
fine-grained annotation schema was proposed to annotate irony
categories, activators and markers in French, English and Italian
language.Theroleplayedbydependency-basedsyntacticfeatures
onironydetectionfromamultilingualperspective(English,Span-
ish, French and Italian) was investigated in [130]. In the case
of satire, in [55] the authors investigated the impact of psycho-
linguisticfeaturesintwodistinctvariantsoftheSpanish(Mexican
and Castilian). Irony detection from a cross-lingual perspective
(Arabic, French and English) was investigated in [131]. Results
showedthat,althoughironyiscontextual,languageandcultural-
depended pragmatic phenomenon, several features are universal
and can be useful for addressing irony detection in languages
whichlackofannotateddata.Inthesameline,in[86]theauthors
presented a set of language independent features that describe
lexical, semantic and usage-related properties of the words in
the tweets. The proposed features were evaluated in a cross-
lingual setting. Results highlighted the complexity of modeling
satirical texts in a cross-lingual setting, due to satire aims at
criticizing social and moral behaviors which often are social and
cultural-dependent.

**3. Multiview informed attention-based models**

In this section we introduce MvAttLSTM, our multi-view in-
formed attentive LSTM model for irony and satire detection in
Spanish. We addressed both tasks as binary classification prob-
lems applying a model based on LSTMs endowed with an atten-
tion mechanism. LSTM is an RNN that uses gating mechanisms
to overcome the problem of the vanishing gradient. This type of
neuralnetworkscancapturelong-rangerelationshipsandhidden
patterns in sequential data. In terms of architecture, the Bidi-
rectional LSTM (BiLSTM) [132] is widely used, which has two
LSTM units processing sequences forward and backward respec-
tively. This property of BiLSTM is useful for language processing
because the meaning of the words in texts can be inferred not
only by previous words, but also considering other words after
them can help to determine their meanings. Moreover, attention
mechanismshaveendowedtheRNNswithapowerfulstrategyto
enhancetheirperformanceandachievebetterresults.Ourmodel
considers multiple representations learned from three distinct
perspectives: linguistic-based representation, universal sentence
encoder-based and contextualized pre-trained embeddings. We
introduce additional knowledge into MvAttLSTM model aiming
at reinforcing linguistics and semantics properties which can
result beneficial for detecting irony and satire. Concretely, our

```
model is compounded by an embedding layer which is fed into a
BiLSTM layer. Later, the hidden states sequence returned by the
BiLSTM is fed into an attention layer. Next, on the output of this
layer are staked two LSTM layers. Finally, we incorporate a feed
forward neural network for final prediction. As explained before,
we inform the model with three additional views. Particularly,
we investigate two different strategies for fusing these views
intoour MvAttLSTM.Inthe nextsubsectionswe presentindetail
the prepossessing carried out on the datasets, the additional
representations, the main parts of the MvAttLSTM’s architecture
and the strategies for informing the model.
```
```
3.1. Preprocessing
```
```
Social media texts, particularly those from Twitter are in-
formal and noisy. The length constraints, and the free writing
style present in this form of online communication provoke that
texts have plenty of grammar and spelling mistakes. Particularly,
length constrains caused that users use shortenings, abbrevia-
tion, homophonic encoding to save characters, and grammar and
spelling misuses such as: character flooding, word repetition and
wrong use of uppercase letters to denote emphasis. Twitter also
offers to the users reserved symbols to mark explicitly important
concepts in tweets ( # hashtag ), to refer o mention other users
( @ mention ), to reply the message of other users ( RT retweets ) or
simplytomarktextsasfavorite( FAV ).Aimingtoinjectemotional
statestoneandbodylanguageintotweets,emoticonsandemojis
became very popular. These symbols are an ultra-concise way to
enrich writing language with visual information. All these issues
impact sentence structure, content, word forms and increase the
difficulty of their automatic processing and comprehension. In
order to mitigate the effects of these problems, in our model we
appliedabasicpreprocessingphaseforcleaningthetexts.Firstly,
weappliedatokenizationprocessonthetweetsbyusingtheTok-
TokTokenizer from NLTK [133]. Later, emoticons, emojis, URLs,
hashtags,mentionsarerecognizedandreplacedbyacorrespond-
ing wildcard which encodes the meaning of these special words.
In the case of hashtag, we replaced the reserved symbol (#) by
the word topic_ and retain the remaining characters. Emoticons
and emojis were replaced by the word emoji_ concatenated with
anintegervalueassociatedtoeachemoji.Wehaveincludedthese
changes in order to reduce the impact of noisy and inconsistent
writingontheprocessingofthetextwiththeFreelingtool[134].
Moreover, we replaced each mention and URL by the words
author_token and url_token respectively. Finally, Twitter-reserved
words like RT (for retweet) and FAV (for favorite) were removed.
It is worth to notice that emoticons and emojis are a valuable
sourceofinformationtotakeintoaccountinsocialmediacontent
analysis [135–141]. Nevertheless, in this work we used emojis
and emoticons to create features for capturing the frequency of
positive emojis, negative emojis and neutral emojis as well as
detecting polarity contradiction between the words and emojis
in the text. In the second stage, and used only to obtain some
linguistic features that were considered in the Linguistic-view ,
a more complex language analysis was carried out. For that,
flooding tokens were normalized allowing the same character to
appearonlytwiceconsecutivelyinatoken(e.g. hooolaaa becomes
hoolaa ). Afterward, tweets were morphologically analyzed with
the FreeLing tool. In this way, for each resulting token, its lemma
and part-of-speech were considered.
```
```
3.2. Addition knowledge to inform the model
```
```
Our MvAttLSTM relies on fusing multiple representations
whicharelearnedfromdistinctperspective.Specifically,welearn
three independent views for each text which are introduced to
```

the model. The first one ( _Linguistics-view_ ) consists in several
linguistics features which have proved to be strong cues for
discriminating both irony and satire. The second one considers a
deepdenseencodingofthemessageusingMultilingualUniversal
Sentence Encoding ( _MUSE-view_ ) [142,143]. And, finally the last
one ( _BERT-view_ ) considers a contextualized pre-trained embed-
ding obtained after a tuning of the BERT model [40]. Next, we
describe the three ways in which each view was learned.

_3.2.1. Linguistics-view_
Hand-crafted features, often linguistic-based, have proved to
beeffectiveforprocessingfigurativelanguage,particularlyincase
of irony, satire and humor [32,62–65]. From our perspective, the
linguistics-based representation is able to capture certain types
of irony, satire and other figurative devices disregarding textual
genres and topics, and it makes this representation content in-
dependent and genre-unbiased. To represent each text, we use
different group of features: stylistic and structural, semantics,
affective,incongruityandpsycho-linguistic.Manyfeaturesareex-
tracted to identify stylistic patterns in the structure of the ironic
orsatiricaltexts(e.g.,typeofpunctuation,length,emoticons,dis-
tribution of nouns, adjectives, adverbs and verbs). Other features
areextractedtoconsideraffectiveinformation(e.g.,polarity,sen-
timents, emotions, attitudes, etc.) by using several word-based
lexiconsresources.Moreover,featuresareextractedforconsider-
ingsemanticspropertiesoftexts(e.g.,co-occurrenceofsynonyms
and antonyms, maximum, minimum and mean of synsets, etc.).
Finally, some features are designated to capture contrast and
opposition in texts (e.g., polarity contrast, semantic incongruity,
etc.). Specifically we use the features proposed in [144,145], the
incongruityfeaturesusedin[146]butusingBabelSenticNet[147]
as default polarity lexicon and including the emotional dimen-
sions and the polarity feature in this resource as other affective
features. BabelSenticNet is an extension of SenticNet [148] to
40 other languages, including Spanish (henceforth we refer the
Spanish version of BabelSenticNet as SenticNet). For more details
about the linguistic features considered in this work please see
Appendix A.

_3.2.2. Task-independent embedding view_
Our second representation aims at encoding the whole mean-
ing of the text into a single dense vector based on deep learning
models.Particularly,invectorswhichcapturerichsemanticinfor-
mationthatcanbeusefulforrecognizingsemanticsproprietiesof
theironicalandsatiricaltexts.Acontextualapproachforcreating
the embedding vectors is proposed in [142], where complete
sentences, instead of words, are mapped into a latent vector
space. The approach provided two variations of Universal Sen-
tence Encoder (USE) with some trade-offs in computation and
accuracy. The first one consists of a computationally intensive
transformer that resembles a transformer network [39], proved
toachieveahigherperformance.Incontrast,thesecondonepro-
videsalightweightmodelthataveragesinputembeddingweights
for words and bi-grams by utilizing of a Deep Average Network
(DAN)[149].TheoutputofDANispassedthroughafeed-forward
neural network in order to produce the sentence embedding.
Both approaches take as input lower-cased strings and output a
512-dimensionalsentenceembedding.Althoughthereareseveral
methods like Doc2Vec [150], Sent2Vec [151], FastText [152] and
InferSent [153] to generate sentence embeddings we used Mul-
tilingual Universal Sentences Encoding (MUSE)^1 [143] which is
an extension of USE trained for 16 languages including Spanish.
The most salience characteristic of this model is that it was

(^1) https://tfhub.dev/google/universal-sentence-encoder-multilingual/
trained using multi-task learning to integrate semantic informa-
tion.Particularly,sentenceembeddingsarelearnedacrossseveral
languagesandusingmultiplesemantictaskslikesentimentanal-
ysis, semantic textual similarity, etc. This enables the learning
process to dynamically accommodate a wide variety of knowl-
edge in a single vector which is interesting to transfer to related
tasks like irony and satire. Based on the MUSE model we trans-
form the texts of the training dataset into dense vectors of 512
dimension (henceforth, _HMUSE_ ). It is important to highlight that
_HMUSE_ is a completely task-independent representation, because
we do not apply any parameters tuning of the model on the
training data.
_3.2.3. Task-dependent embedding view_
The transformer-based neural network architectures [39]
paved the way for training robust and contextual-aware lan-
guage models in an unsupervised manner. The use of these pre-
trained contextualized models have been widely spread through
BERT [40] and BERT’s family architectures [41–44]. To study the
linguistic and semantic nuances of irony and satire in Spanish,
we decide to incorporate BERT as another representation (view).
BERTreliesonbidirectionalrepresentationfromtransformersand
achieves the state of the art for contextual language modeling
and contextual pre-trained embeddings. This model is trained on
a large text corpus and then used for downstream NLP tasks.
While other word embedding like Word2Vec [154], Glove [155]
and FastText [152] are context-free models that produce a single
wordembeddingforeachwordinthevocabulary,BERTcomputes
a representation of each word that is based on the other words
in the context. It was built upon recent works in pre-training
contextual representations, such as ELMo [156] and Universal
Language Model Fine-tuning (ULMFiT) [157], and is deeply bidi-
rectional.BERTrepresentseachwordusingbothitsleftandright
contexts. Moreover, it is possible to fine-tune BERT for many
downstream NLP tasks, including the tasks we are interested in.
This goal can be achieved by removing the language modeling
output layer (masked word prediction) and replacing it with a
new layer appropriate for the target task (in our case, binary
classification). Particularly, in this work we use the pre-trained
multilingual versions of BERT^2 (mBERT, henceforth) and carried
out a fine-tuned on it, using the training datasets of irony and
satire. Our idea is not to use this model for as a classification
method;instead,weconsidereditfortherepresentationpurpose.
For fine-tuning mBERT, we add a layer that receives as the
inputthevectorinthefirstposition(the _CLS_ token).Onthislayer,
westackedanoutputlayerthatmakesthefinalpredictionforthe
targeted task. For that purpose, we follow the strategy proposed
in ULMFiT [157]. For each layer of mBERT, a different learning
rate is set up, increasing it using a multiplier while the neural
network gets deeper. This multiplier increases 0.1 points from
a layer _Li_ to another _Li_ + 1. We use this dynamic learning rate to
keep most information from the pre-training at shallow layers
and biasing the deeper ones to learn about the specific tasks.
Forallcorpora,thesamehyperparameterswereused.Concretely,

### we defined the batch _ size = 32 and the sequence length was

```
limited to 50 tokens. The optimizer used is Adam [158] with an
```
### initial learning rate of 0.001,β 1 = 0 .9 andβ 2 = 0 .999 and a

### w eight _ decay = 0 .01. The model was trained during 15 epochs

```
and using the ModelCheckpoint callback for obtaining the model
that has achieved the best performance on the validation subset.
After tuning mBERT, we pass again the training dataset, but
this time, we get the deep representation of each text of the
training dataset (henceforth, HBERT ). This view is task-dependent
because we refine the parameters learned by mBERT in order to
capture semantics and pragmatics characteristics, which results
crucial for understanding and recognizing ironic and satirical
intents.
```
(^2) https://huggingface.co/bert-base-multilingual-cased


_3.3. MvAttLSTM model_

Let us describe the architecture of the MvAttLSTM model. We
give details about each layer, starting from the embedding layer
to the loss function.

_3.3.1. Embbeding layer_

### In this layer each wordw i is map into a highly dimensional

feature space for capturing the meaningful semantic and syn-
tactic information. Given an input text _T_ which consists of at

### most N wordsw i , where i ∈ [ 1 , N ]. For each word into T , we

### examine the embedding matrix E ∈ RB × d , where B is the length

of the vocabulary, and _d_ is the dimension of word embedding
vectors. The matrix _E_ can be initialized randomly or by means
of a word embedding matrix. In this work we decided to ini-
tialize the embedding layer with context-free pre-trained word
representations. For that, we learned the embedding matrix _E_ by
using the FastText model trained on the Spanish Billion Words
Corpus^3 and an in-house background corpus of 9 millions of
Spanishtweets.Weaimtojoinbothcorporaforobtainingrobust
word representations taking advantage of the peculiar writing
style used in Twitter. For training the FastText model we used
the setting reported in [152], except for the value of the vector
size, which was defined as a 300-dimensional. In this layer, each

### wordw i is transformed into a vector xi ∈R d :

### H^0 = Embbeding ( E , M ) (1)

Thus, every text _T_ can be converted in a sequence of vectors, in

### the form of a 2d-matrix H^0 = [ x 1 , x 2 , x 3 ,..., xN ] T with shape

### N × d. The matrix H^0 is given as input to the next layer. It is

worth noting that in the model the weights in _E_ are fixed. We
aim at making the model to be trained faster and mitigate the
impact of the overfitting due to the reduction of parameters that
must be learned. Moreover, we consider that the recurrent and
attention layers are feasible to take advantage of the semantic
and syntactic properties of the vectors in _E_ for classifying irony
and satire.

_3.3.2. BiLSTM layer_
After passing the sequence of word _T_ to the Embedding layer,

### eachwordw i isencodedbyavector xi whichcapturestheseman-

### tic and syntactic properties ofw i out of context. In other words,

### the representation of eachw i is independent of the other words

in the text _T_. In this layer, a new representation for each word
is learned by summarizing the contextual information, previous
and after to the word in the text. For achieving this goal we
use a BiLSTM layer. This, type of neural network consists of two
LSTM units which process the sequential input in both directions
forward and backward simultaneously.

### H^1 = BiLSTM ( H^0 ) (2)

### The output of this layer is a sequence of hidden states H^1 =

### [ h^11 , h^12 ,..., h^1 N ]where each h^1 i ∈R^2 × dh is the concatenation of

the hidden state of each LSTM (right and left), specifically, the

### hi = [

### −→

### h^1 i ,

### ←−

### h^1 i ], and dh is the number of hidden neuron into the

LSTM unit. Standard LSTM receives sequentially (in a left to right
order) at each time step a word vector _xi_ and produces a hidden
state _hi_. For that, this neural network relies on a cell of memory
and a gating mechanism consisting of an input gate, forget gate,
and output gate. These gates help to determine whether the
information in the previous state should be retained or forgotten
inthecurrentstate.Hence,thegatingmechanismhelpstheLSTM

(^3) https://crscardellino.github.io/SBWCE/
to cope with long-term information preservation. Each hidden
state _hi_ is determined as follows:

### It =σ( Wixt + Uiht − 1 + bi ) (3)

### Ft =σ( Wfxt + Ufht − 1 + bf ) (4)

### Ot =σ( Woxt + Uoht − 1 + bo ) (5)

### C ̃ t =σ( Wuxt + Uuht − 1 + bu ) (6)

### Ct = it ⊙ C ̃ t + Ft ⊙ Ct − 1 (7)

### ht = Ot ⊙ tanh ( Ct ) (8)

```
Whereall W ∗, U ∗and b ∗areparametersoftherecurrentlayer
which are learned during the training phase and the xt is the
pre-trained vector of the word in the time-step t and it is not
```
### trained in the model. The operatorσis the sigmoid function and

### theoperator⊙standsforelement-wisevectormultiplication.The

### It , Ft and Ot are the input, forget and output gates in the time

### step t −1whereas C ̃ t , Ct and ht arethenewcell,theupdatedcell

```
memory and the final hidden state in the time step t. Notice that
the BiLSTM initial hidden states and cells memory are set to 0 in
both directions
```
### −→

### c^10 =

### ←−

### c 01 =⃗0 and

### −→

### h^10 =

### ←−

### h^10 =⃗0. We highlight

```
this detail because we use these states as the way to incorporate
additional information into the MvAttLSTM model.
```
```
3.3.3. Attention layer
TheBiLSTMLayerhastwomajorproblems.Sincethemeaning
of the message cannot be encoded in one fixed-size vector, there
is some information loss. Hence, the performance of this type of
models for representation learning decreases when the length of
inputs become large. Another concern is that LSTMs aggregate
information word-by-word in sequential order, but there is no
explicit mechanism to make inferences over the structure and
modelingrelationsamongtokens.Toovercometheselimitations,
```
### the output of the BiLSTM Layer H^1 ∈ R^2 × dh × N is fed into an

```
Attention Layer. This layer helps BiLSTM in deciding which parts
of the sequence pay more interest. In this work, we investigate
the performance of two attention mechanisms in our model:
self-attention and multi-head attention [39].
Self-attention mechanisms can capture the explicit and latent
relations among words beyond their sequential order. While at-
tention mechanism [159] allows the outputs for attending some
parts of the inputs. Self-attention also allows the inputs for in-
teracting each other, hence amplifying the importance of each
one plays in determining the meaning of others. Moreover, is
it beneficial for discovering word relations which can be crucial
for understanding ironic and satirical texts such as, oppositions
and incongruities. Given the matrices A , B and C , mathematically
self-attention is formulated as follows:
```
### Att ( A , B , C )= Attention ( AWQ , BWK , CWV ) (9)

### Attention ( Q , K , V )= softmax (

#### QKT

### √

```
dk
```
#### ) V (10)

### Where WQ ∈R d , WQ ∈R d , WQ ∈ R d are the projection

```
matrices for the query Q , key K and value V. In the case of self-
attention, the matrices A , B and C are the same. Thus, given the
output H^1 of the BiLSTM layer, the new sequence of weighted
```
### hidden states H^2 ∈R^2 × dh × N which is the output of the Attention

```
layer is computed by Eq. (11) as:
```
### H^2 = Att ( H^1 , H^1 , H^1 ) (11)


In [39] another attention mechanism was introduced. The
multi-head attention mechanism uses multiple individual atten-
tionfunctions(heads)forobtainingdifferentcontextsandpaying
attention simultaneous to distinct aspects in the sequences. This
can jointly pay attention to information from different repre-
sentation sub-spaces at different positions. Like in self-attention,
the attention function takes as input a matrix for the query _A_ , a
matrix for the keys _B_ and a matrix for values _C_. The multi-head

### attention model first transforms A , B and C intoCsub-spaces,

with different, trainable linear projections:

### MultiHead ( A , B , C )=[ head 1 , head 2 ,..., headr ]∗ W^0 (12)

### headc = Attention ( AWcQ , BWcK , CWcV ) (13)

### Where WcQ ∈R d × dk , WcK ∈R d × dk , WcV ∈R d × d vare projec-

### tion matrices for the inputs A , B , C with respect to headc , and

### W^0 ∈ R r × dk × d. The parameter r is the number of heads for

### the multi-head attention mechanism; and headc ∈R N × dk is the

output of the _c_ th head. Notice that, for each _headc_ , the weights
of _W
Q_

### c , W

```
K
```
### c , W

_V
c_ are independently learned during the training
phase. The attention for each head _c_ (see Eq. (13)), like in self-
attention, is computed by the formula in Eq. (10). Thus, given
the output of the BiLSTM layer _H_^1 , the output of the multi-head
attention is computed as follows:

### H^2 = MultiHead ( H^1 , H^1 , H^1 ) (14)

_3.3.4. LSTM layers_
Even when, it is not theoretically clear what is the additional
power gained by the deeper recurrent architectures, it was ob-
servedempiricallythatadeepLSTMworksbetterthanshallower
ones in some tasks [100,160]. Taking this into account, on the
output _H_^2 of the Attention Layer we stacked two layers of LSTMs
to deep contextualize the previously learned representation. This
means that the output of the first LSTM layer is given as input to
the second one. The two LSTM layers are defined as follows:

### H^3 = LSTM ( H^2 ) (15)

### H^4 = LSTM ( H^3 ) (16)

### Where H^3 ∈R dk^1 N , H^4 ∈R dk^1 N are the outputs of the first and

second LSTM layer and _dk_ 1 is the number of hidden neurons into
LSTM cells. The last LSTM layer output the hidden representation
of the text. Particularly, we only consider the last hidden state
( _h_^4 _N_ ) into the matrix _H_^4. Let us redefine it as _h_^4 _last_ henceforth. Like
in the BiLSTM layer, the initial hidden state and cell memory of

### both LSTMs are set to 0, h^30 = c^30 =⃗0, and h^40 = c 04 =⃗0.

_3.3.5. Fusion strategies_
Our model aiming at improving irony and satire detection in
Spanish by incorporating multiple views into the MvAttLSTM.
For this purpose, we investigate two strategies for passing the
views to the model. The first one, _Early Fusion_ method, aiming
at enriching the representation learned by the LSTMs with ad-
ditional knowledge using the last dense layers. The second one,
_Contextual Fusion_ method, which aims to condition the learning
process of the LSTMs with prior knowledge injected in the initial
cell memory. Next, we give details about both strategies.

_Early Fusion_
Themainideabehindthisstrategyistoseparatelylearndiffer-
ent features spaces from the training data. These feature spaces
(views) capture distinct characteristics of the same texts. We
aim to jointly use these representations to retain discriminant
informationwhilereducingtheredundantone.Theoverallarchi-
tecture of the MvAttLSTM with _Early fusion_ is showed in Fig. 1.

```
Fig. 1. MvAttLSTM: Multi-view informed attentive LSTM deep neural network
using an early fusion strategy.
```
```
Firstly, let us define HLing , HMUSE and HBERT as the Linguistic-
view, MUSE-view and BERT-view respectively (see Section 3.2).
These views differ from each other in the way by which were
learned and the number of features used for encoding the text.
Thus, we pass each view to a dense layer in order to reduce and
unify the views’ dimensionality using the Eq. (17), (18), (19):
```
### gl ( HLING )=σ( WlHLING + bl ) (17)

### gm ( HMUSE )=σ( WmHMUSE + bm ) (18)

### gb ( HBERT )=σ( WbHBERT + bb ) (19)

### Where Wl , Wm , Wb and bl , b , bb are parameters of the model

### to be learned during the training process, andσis the sigmoid

```
function. After having reduced representations for each view
```
### ( gl , gm , gb ),thenweconcatenatethemwithadeeprepresentation

```
learned by the attentive LSTM based architecture h^4 last (Eq. (20)).
Later,themergedrepresentationdenotedas F 0 isfedintoadense
layer with sigmoid activation for fusing all views into a new
non-linear space using Eq. (21). Finally, the output of this layer
denoted as F 1 is a multi-view encoding of the texts, and it is fed
into a feed-forward neural network for the final classification of
the texts in ironic vs. non-ironic or satirical vs. non-satirical.
```
### F 0 = Concat ( gl , gm , gb , h^4 last ) (20)

### F 1 =σ( W^0 F 0 + b^0 ) (21)

```
Contextual Fusion
In this strategy, we enrich our MvAttLSTM with additional
external knowledge to take advantage of the initial memory
cell in the LTSMs. We experiment with a strategy similar to
the conditional encoding model introduced in [161] for the task
of recognizing textual entailment and applied later in [82] for
modeling conversation context for improving sarcasm detection.
Conversely, to the approach presented by Ghosh et al. [82], we
do not learn contextual information by using LSTMs, instead, we
learn independently three distinct views with the aim to capture
syntactic, semantic, and pragmatics aspects used in ironical and
satirical texts. In Fig. 2 is showed the overall architecture of our
MvAttLSTM model using the Contextual Fusion strategy. As can
be observed, the learning process of each LSTM is conditioned to
prior information passed to the initial memory cell.
Like in the Early fusion strategy, firstly each view is fed into
a dense layer with non-linear activation, specifically using the
```
### Eq. (17), (18), (19). Later, each reduced representation ( gl , gm , gb )


**Fig. 2.** MvAttLSTM: Multi-view informed deep attentive LSTM neural network
using contextual fusion strategy.

isusedtoinformtheLSTMintheMvAttLSTM.Theinitialmemory
cell and hidden states of the LSTMs are used as input to pass the
prior knowledge of each view as defined in Eq. (22), (23), (24).
Where _c_ 01 and _h_^10 aretheinitialmemorycellsandhiddenstatesof
the BiLSTM. And, _h_^30 , _c_ 03 , _h_^40 and _c_ 04 are the initial memory cell and
hiddenstatesofthesecondandlastLSTMrespectively.Theorder
in which each view is assigned to the LSTMs was empirically
defined. We decide to introduce low-level linguistic features for
reinforcing the BiLSTM layer which aims at capturing language
generalization. In the second LSTM, we propose to introduce the
MUSE-view for incorporating a high-level semantic representa-
tion to encode the global meaning of the text. Finally, in the last
LSTM, we introduce the BERT-view to incorporate semantics and
pragmatics abstractions useful for the task to solve, considering
thatinthisviewtheBERTmodelistunedusingthesametraining
data available for the task. This introduces a task-dependent bias
in the language representation learned by the original model:

### c 01 = h^10 = gl ( HLING ) (22)

### h^20 = c 02 = gm ( HMUSE ) (23)

### h^30 = c 03 = gb ( HBERT ) (24)

Notice that in this case, the final multi-view representation of
the text _F_ 1 is the same that the last hidden state of the last LSTM

### layer h^4 last in our MvAttLSTM, hence F 1 = h^4 last. And, we pass

this representation into the feed forward neural network for the
classification of the texts in ironic vs. non-ironic or satirical vs.
non-satirical.

_3.3.6. Feed-forward layer for final classification_
Forachievingthefinalclassificationwefedthemulti-viewen-
coding of the text _F_ 1 into a Dropout layer to prevent the model’s
over-fitting. Subsequently, the output of the Dropout layer _F_ 2 is
passed to a dense layer with ReLU activation, and finally, the
output of this layer _F_ 3 is given as input to another dense layer
withtwoneurons,butthistimewiththe _softmax_ functionforthe
prediction:

### F 2 =Dropout( F 1 ) (25)

### F 3 = max (0, W^2 F 2 + b 2 ) (26)

### O = softmax ( W^3 F 3 + b 3 ) (27)

The MvAttLSTM model can be trained in an end-to-end way
by the back-propagation method, and we use categorical cross-
entropy as the loss function. This function can be observed in

### Eq. (28), whereDis the dataset,Lis the loss function, f is our

### model parameterized byθandG= { 1 , 0 }is the set of labels in

```
the task.
```
### L(θ)=ED[L( f ( x ,θ), y )]=−

#### 1

```
n
```
## ∑ n

```
i = 1
```
## ∑|G|

```
j = 1
```
### yij ∗log( f ( xi ,θ) j )w j (28)

**4. Experiments and results**

```
4.1. Datasets description
```
```
In order to validate our proposed model for irony and satire
detection in short texts written in distinct Spanish variants, we
usedthreecorpora,oneforironydetection,andtwoforsatirede-
tection.Moreover,wealsotestedtherobustnessofourmodelon
humorrecognitiononanothercorpus.Theyhavebeenextensively
used with the aim of training and evaluating state-of-the-art
systems for irony, satire and humor detection in Spanish.
Irony Corpus
For what concerns irony detection, we decided to use the
corpus proposed in the IroSvA’19 shared task [56]. This is the
first public available corpus for irony detection in Spanish. The
IroSvA’19 sharedtask,framedintheIberianLanguagesEvaluation
Forum (IberLEF’19)^4 and co-located within SEPLN 2019^5 aimed
at investigating whether a short message, written in Spanish, is
ironic or not within a given context. For that, three corpora with
shorttextsfromSpain,MexicoandCubawereproposedwiththe
purpose of exploring the way irony changes in Spanish variants.
In particular, the Castilian and Mexican corpora consist of ironic
tweets about 10 controversial topics for Spanish and Mexican
users. In the case of the Cuban corpus, it consists of ironic news
comments which were extracted from 113 controversial news
aboutsocial,economic,andpoliticalissuesconcerningtheCuban
people. It is worthy to notice that, for each text a context is
provided,consistingofa shortdescriptionaboutthetopic,which
defines its scope. The distribution of the texts is showed in
Table 1.
As can be observed in Table 1 all subcorpora are composed of
3000 texts split into 2400 and 600 texts for training and testing
respectively. The training set is separated into 800 ironic and
1600 non-ironic texts, whereas the testing partition is divided
into200ironicand400non-ironictexts.Noticethatbothtraining
and testing sets maintain the ratio of 2/3 vs. 1/3 between non-
ironic and ironic text. In order to assess the performance of the
systems,theevaluationmetricsusedbytheorganizerswerepre-
cision (P), recall (R), and F1 score. These metrics were calculated
perclassandmacro-averaged.Duetotheimbalancebetweenthe
non-ironic and ironic classes, the macro-averaged F1 score was
used as the overall metric to rank the participating systems.
Satire Corpora
For satire detection, we used the corpora proposed in [55]
and[86].Bothcorporawerecreatedusingaself-annotationstrat-
egy.Specifically,Barbierietal.[86]retrievedtweetsfrompopular
satiricalnewsaccountsandfromlegitimatenewssourcesinthree
languages: Spanish, English, and Italian. In this work, we were
interested in the Spanish subset of this corpus, and we refer to
this as Barbieri’15-es henceforth. The Spanish tweets (Castilian
variant) were gathered from two satirical Twitter’s accounts El
Mundo Today and El Jueves whereas non-satirical tweets were
retrieved from the legitimate newspaper Twitter’s accounts El
Mundo and El Pais. Later, a shallow cleaning process was carried
out on data for filtering those tweets that were not relevant
to satire analysis. As can be observed in Table 2, the corpus is
```
(^4) https://sites.google.com/view/iberlef-
(^5) [http://www.hitz.eus/sepln2019/?language=es](http://www.hitz.eus/sepln2019/?language=es)


```
Table 1
IroSvA’19 distribution for ironic and non-ironic classes.
Corpus Variant Training Testing
Non-Ironic Ironic Total Non-Ironic Ironic Total
Castilian (es) 1600 800 2400 400 200 600
IroSvA’19 Mexican (mx) 1600 800 2400 401 199 600
Cuban (cu) 1600 800 2400 400 200 600
```
composedof10888uniformlydistributedin5444satiricaltweets
5444 non-satirical ones.
Thecorpusintroducedin[55]wasguidedbythesamemethod-
ology presented in [86]. The most salience difference relies on
the study of satirical tweets in two variants of the Spanish. Par-
ticularity, tweets from Mexican and Castilian Twitter’s accounts
wereretrieved.ForinvestigatinghowsatireisrealizedinMexican
tweets, data from four Mexican Twitter accounts were retrieved.
The satirical tweets were obtained from _El Deforma_ and _El Dizque_
satiricalaccountswhereasthenon-satiricaltweetsweregathered
from legitimate newspaper accounts _El Universal_ and _Excelsior_.
We refer to this subset of data as _Salas’17-mx_ henceforth. The
tweetsintheCastiliancorpusof[55], _Salas’17-es_ henceforth,were
retrieved using the four Twitter’s accounts proposed in [86]. It is
important to note that even when the Twitter’s accounts used to
obtain the tweets were the same, the tweets in each collection
are different.
An automatic cleaning process was carried out on the data.
Specifically, retweets, duplicates, tweets only with URLs, and
tweets written in a language other than Spanish were removed.
Moreover,amanualinspectionwasperformedinordertoensure
that the tweets obtained were relevant for satire detection. In
Table 2 can be observed that both corpora contain 5000 tweets,
which are uniformly distributed in 2500 satirical and 2500 non-
satirical ones. The different characteristics of the _Barbieri’15-es_
and _Salas’17-es_ will allow us to validate the robustness of our
_MvAttLSTM_ model.

_Humor Corpus_
For further investigating the robustness of our model we de-
cided to evaluate it on humor recognition in Spanish. We consid-
ered the corpus proposed in the _HAHA’19_ shared task [162,163]
organized at _IberLEF’2019_ and co-located within _SEPLN 2019_. Two
subtaskswereproposed,oneforhumorbinaryclassification( _Hu-
mor Recognition_ )andanotherforpredictinghowfunnyisatweet
into 5-star ranking ( _Funniness Score Prediction_ ), considering that
the tweets present humorous content. The organizers provided
a human-annotated corpus of 30000 Spanish tweets separated
into 24000 for training and 6000 for testing. The training subset
consistsof9253humorousand14747non-funnytweets,whereas
the testing subset consists of 2342 humorous tweets and 3658
non-funny ones. In Table 3 we summarize the distribution of the
tweets within HAHA’19. Taking into account the scope of this
work,weareonlyinterestedinthefirstsubtask.Ascanbenoted,
inbothtrainingandtestingsubsetsthedistributionoftheclasses
areslightlyunbalanced,henceadifficultyisaddedtothelearning
algorithm.Theperformancemetricsusedtoranktheparticipated
systems in the _Humor recognition_ subtask were F1 score for the
_humorous_ class and accuracy _(Acc)_.

_4.2. Experimental setting_

We use the same architecture for all tasks, but we calibrated
thehyper-parametersindependentlyforeachcorpus.Specifically,
we defined the number of hidden neurons in the BiLSTM layer
to 64, the number of hidden neurons in the last two LSTM
layer to 128, the maximum number of epochs and length of the

### sequence ep = 50 and N = 50 respectively. For the remain-

der of hyperparameters, we experimented with distinct values.

```
Specifically, we defined the search space as follows: batch size
```
### batch ∈ [ 32 , 64 , 128 , 265 ], dropout dp ∈ [ 0. 25 , 0. 30 , 0. 35 , 0. 4 ],

### optimizer update rules op ∈( adam , rmsprop ), learning rate lr ∈

### [ 1 × 10 −^2 , 1 × 10 −^3 , 1 ×^10 − 4 ].Whenthemodelusesmulti-head

### attentionweevaluateddistinctnumberofheads h ∈[ 2 , 4 , 8 , 16 ].

```
Inanintenttopreventtheoverfittinginthetrainingstep,anearly
stopping with the patience of 10 epochs was used as a stopping
criterion. We explored the search space by means of the Grid
Search strategy. Analyzing the best hyperparameters obtained
for each corpus and model, we observed that our models are
sensitive to the hyperparameter setting. Particularly, we noted
```
### thatthelearningrate lr = 1 × 10 −^2 achievedthebestperformance

```
across all corpora. However, the remainder of the hyperparam-
eters has a distinct behavior. Roughly speaking, we appreciated
```
### that the M v AttLSTMselfContextual and M v AttLSTMmultiContextual

```
are the most sensitives models. One possible reason for that
is the small number of ironic examples in each corpus which
makesthemodelgeneralizationmorecomplex.Inthecaseofthe
```
### M v AttLSTMselfEarly model, it was observed that it performs well

### on large corpora using short batches ( batchsize = 32) whereas

### M v AttLSTMmultiEarly requires longer batches ( batchsize = 128).

```
Also, we observed that 4 heads of attention were enough to
achieve good results. Concerning the optimizer, generally, the
Adam ruleobtainedthebestresultin9settingsoutof16.Thebest
hyperparameters for each corpus and model are summarized in
Appendix B.
In order to evaluate the performance of the distinct settings
of our model, we define a baseline method (Bert-baseline). Con-
cretely, we use the mBERT method fine-tuned on each corpus
separately. For that, we adopt the same hyperparameter and
tuning strategy proposed in Section 3.2.3.
Regarding the Linguistic-view , we experimented with distinct
views to investigate whether some groups of features are more
feasiblethanotherstodetectironycues.Inthissense,wedefined
threeviewsforconsideringaffectiveinformation: Aff_All, Aff_Emo,
Aff_App. In Aff_All we considered all features related to polarity,
emotions (categorical, and dimensional), and attitudes. Whereas
in Aff_Emo we only used those features related to emotions, and
in Aff_App we only use the attitude words. The features that
capturepolarityoppositionswereincludedinthegroup Contrast.
We evaluated two groups ( LIWC, Sverb ) based on the psycho-
linguistic dimensions in the LIWC dictionary and the semantic
classes of verbs in the ADDESE lexicon^6 respectively. Moreover,
other groups of features were obtained by using a feature se-
lection method, specifically the Wilcoxon rank-sum test [164]
wereexplored.Withthisstatisticaltest,thefeatureswereranked
considering their p -value, and three groups were defined. In the
groups W_64 , W_128 and W_All we considered the subsets of
64 and 128 best-ranked features and all features with p -value
```
### ≤ 0 .05.Finally,agroupwithallthelinguisticfeatures LingAll was

```
considered.
```
```
4.3. Results in irony detection in spanish variants
```
```
Inthissection,wepresentanexhaustiveevaluationofdistinct
settings of the MvAttLSTM model in the task of irony detection.
```
(^6) [http://adesse.uvigo.es/data/clases.php](http://adesse.uvigo.es/data/clases.php)


```
Table 2
Distribution for satirical and non-satirical classes in Salas’17 and Barbieri’15 datasets.
Corpus Variant Data
Non-Satirical Satirical Total
```
```
Salas’17 Castilian (es)Mexican (mx)^250025002500250050005000
```
```
Barbieri’15 Castilian (es) 5444 5444 10888
```
```
Table 3
HAHA’19 distribution for humorous and non-humorous classes.
Corpus Language Training Testing
Non-Humorous Humorous Total Non-Humorous Humorous Total
HAHA’19 Spanish 14747 9253 24000 3658 2342 6000
```
Our first experiment aimed at investigating the impact of differ-
ent types of linguistic views on the model. For that, we analyzed
what subsets of features are most relevant to irony detection.
The second aspect that we considered relevant to explore was
the impact of the fusion strategies to inform the model ( _Early vs.
Contextual_ )andtheattentionmechanismusedbytheMvAttLSTM
model ( _self vs. multi-head_ ). Lastly, we investigated the impact of
each proposed view. For that, we ignored one view and fed the
other two into the MvAttLSTM model.
To evaluate the effectiveness of our proposal in each exper-

iment we computed the F1 score for the two classes ( _F_ (^1) _iro_ and
_F_ (^1) _no_ − _iro_ ), along with their macro-averaged and micro-averaged
versionsofF1( _F_ (^1) _Micro_ and _F_ (^1) _Macro_ ).Wesplitthetrainingdatainto
80% and 20% for training and validation purposes and evaluated
the generalization of our model on the official test provided by
the organizers. The results obtained for _IroSvA’19_ corpus on the
test dataset in the three Spanish variants are shown in Table 4.
We only included the results using the _Contextual fusion_ strategy
for the Castilian (es), the Mexican (mx) and the Cuban (cu) vari-
ants due to another fusion strategy achieved worse results in the
three variants.

### ItcanbeobservedinTable4thatthemodel M v AttLSTMmultiContextual

### obtained slightly better results than M v AttLSTMselfContextual in the

three variants ( _es, mx_ and _cu_ ) for all the evaluation metrics. Con-
cretely,fortheCastilianvariant,thebestresultswereobtainedby

### M v AttLSTMmultiContextual using all views, but in the case of Linguistic-

_view_ only considering emotional _Aff_Emo_ or attitudinal features

### Aff_App. Moreover, the model M v AttLSTMContextualself achieves com-

petitive results but considering the views _W_128_ or _W_64_. Re-

### gardingtheMexicanvariant,bothmodels M v AttLSTMmultiContextual and

### M v AttLSTMselfContextual obtained the best results when the Aff_All

and _Aff_Emo_ views are used respectively. Also, it is important to
notice that the second better results for each model are achieved
when the views _LIWC_ and _Aff_App_ are considered. In the case of
the Cuban variant, both models obtain similar results. However,

### M v AttLSTMmultiContextual using all linguistic features LingAll slightly

### outperform M v AttLSTMmultiContextual with the linguistic view Aff_All.

To sum up, we found that some subsets of features are the
most relevant in our model for irony detection in the three vari-
ants; particularly, those related to affective information such as
_Aff_Emo_ and _Aff_App_. This fact indicates the discriminatory prop-
erty of the emotional dimensions in SenticNet and the emotional
categories in Spanish Emotions Lexicon (SEL) [165]. Also, the
attitude-based features obtained from Appraisal Lexicon (LAM)
[166] were relevant. This result is in line with the findings pre-
sentedin[75]whichinvestigatedtheroleofaffectiveinformation
in irony detection using machine learning models. Furthermore,
we found that the _Bert-baseline_ method performs significantly
worsethantheMvAttLSTMmodelinthethreevariants.Onepos-
sible explanation for that is the small number of ironic examples
in the training dataset that make more complex the learning
process.

```
In a second direction, we investigated the importance of each
proposed view ( Bert-view, Muse-view and Linguistic-view) on the
performance of MvAttLSTM. For that, we evaluated the model
ignoring one view and including the remaining two. In this ex-
periment, the Linguistic-view ( Ling ) represents the subsets of
```
features that achieved the best _F_ (^1) _Macro_ (see Table 4). Notice that
_Ling_ is different for each MvAttLSM setting and dataset. The
results obtained are summarized in Table 5.
As can be shown in Table 5, the best _F_ (^1) _Macro_ in all corpora
was obtained when all views were used together. This fact con-
firms that informing our model with the proposed views helps
the model to detect irony. However, we observed that for the
Castilian variant, ignoring _Muse-view_ caused the most significant

### drop in performance of M v AttLSTMselfContextual whereas omitting the

### Bert-view producedtheworseperformancein M v AttLSTMmultiContextual.

### In the case of the Mexican variant, both settings of M v AttLSTM

```
achieved the worse performance when the Muse-view was re-
movedfromthemodel.However,fortheCubanvariant,wefound
that the model drops its performance when the Linguistic-view
wasomitted.AnalyzingtheresultsinTables4and5together,the
linguistic view (LingAll) was found to produce a lower F1-macro
than when no linguistic features are introduced in the model.
In this sense, we considered that a deeper analysis would be
necessary to explain the reasons for the negative result achieved
when including all the linguistic features whether it is due to
noisy features or the fusion strategy used to feed this view into
the model. Further efforts need to be made for investigating
why the attention mechanisms ( self vs. multi ) attend different
linguistic views for obtaining better effectiveness.
Following, we present a comparison of our best results on
the three corpora with other state-of-the-art systems. In Table 6
we show how the results of the participating systems in the
IroSVA’19 shared task ranked according to the official evaluation
```
measure _F_ (^1) _Macro_ average. It is important to highlight that the
participants were not restricted to submit the same system for
each corpus. Thus, the F1-AVG means the average of the re-
sults of the team instead of evaluating the performance of one
model on the three corpora. Our best model for the Castilian

### variant esM v AttLSTMmultiContextual outperforms the results achieved by

```
ELiRF_UPV [45,58] and CIMAT [59] on the Castilian and Cuban
corpora. However, our model drops its performance on the Mex-
ican corpus. Regarding our best model for the Mexican vari-
```
### ant mxM v AttLSTMmultiContextual , it outperforms the results obtained by

### ELiRF_UPV and CIMAT on all corpora. The cuM v AttLSTMmultiContextual

```
model achieved better results than ELiRF_UPV and CIMAT on the
Cuban corpus but drooped its effectiveness on the Castilian and
Mexican variants. It is important to remark that ELiRF_UPV is
basedonadeeplearningmodel;particularlyitproposedasimpli-
ficationoftheBERTmodel,andCIMATproposedacombinationof
deep learning-based representations with n-gram features. From
an overall point of view, our proposed models are placed in the
```

**Table 4**
MvAttLSTM for irony detection in _IroSvA’_.

_Views F_ (^1) _iro F_ (^1) _no_ − _iro F_ (^1) _Micro F_ (^1) _Macro F_ (^1) _iro F_ (^1) _no_ − _iro F_ (^1) _Micro F_ (^1) _Macro
M_ v _AttLSTMContextualself IroSvA’19-es M_ v _AttLSTMContextualmultiheadIroSvA’19-es
Bert+Muse+Aff_All_ 0.487 0.821 0.734 0.654 0.596 0.835 0.766 0.
_Bert+Muse+Aff_Emo_ 0.619 0.815 0.751 0.717 **0.668 0.842 0.786 0.**
_Bert+Muse+Aff_App_ 0.538 0.82 0.741 0.679 **0.651 0.835 0.776 0.**
_Bert+Muse+Contrast_ 0.59 0.833 0.762 0.712 0.587 0.83 0.759 0.
_Bert+Muse+LIWC_ 0.549 0.801 0.724 0.675 0.626 0.831 0.767 0.
_Bert+Muse+SVerb_ 0.576 0.798 0.726 0.687 0.627 0.824 0.761 0.
_Bert+Muse+W_64_ **0.629 0.832 0.769 0.731** 0.583 0.834 0.762 0.
_Bert+Muse+W_128_ **0.656 0.835 0.777 0.746** 0.642 0.82 0.761 0.
_Bert+Muse+W_All_ 0.595 0.811 0.742 0.703 0.55 0.808 0.731 0.
_Bert+Muse+LingAll_ 0.553 0.802 0.726 0.678 0.589 0.831 0.761 0.
_Bert-baseline_ 0.302 0.741 0.594 0.521 0.302 0.741 0.594 0.
_M_ v _AttLSTMContextualself IroSvA’19-mx M_ v _AttLSTMContextualmultiheadIroSvA’19-mx
Bert+Muse+Aff_All_ 0.516 0.79 0.708 0.654 **0.647 0.817 0.759 0.**
_Bert+Muse+Aff_Emo_ **0.642 0.821 0.761 0.732** 0.508 0.785 0.701 0.
_Bert+Muse+Aff_App_ **0.644 0.794 0.739 0.719** 0.585 0.774 0.708 0.
_Bert+Muse+Contrast_ 0.601 0.812 0.744 0.706 0.506 0.792 0.708 0.
_Bert+Muse+LIWC_ 0.506 0.792 0.708 0.649 **0.611 0.826 0.759 0.**
_Bert+Muse+SVerb_ 0.564 0.765 0.694 0.664 0.596 0.828 0.759 0.
_Bert+Muse+W_64_ 0.529 0.788 0.708 0.659 0.515 0.786 0.703 0.
_Bert+Muse+W_128_ 0.567 0.777 0.706 0.672 0.591 0.591 0.689 0.
_Bert+Muse+W_All_ 0.512 0.79 0.706 0.651 0.599 0.790 0.724 0.
_Bert+Muse+LingAll_ 0.606 0.809 0.743 0.707 0.469 0.808 0.718 0.
_Bert-baseline_ 0.293 0.771 0.611 0.532 0.293 0.770 0.611 0.
_M_ v _AttLSTMContextualself IroSvA’19-cu M_ v _AttLSTMContextualmultiheadIroSvA’19-cu
Bert+Muse+Aff_All_ **0.604 0.807 0.741 0.706** 0.534 0.796 0.716 0.
_Bert+Muse+Aff_Emo_ 0.574 0.809 0.736 0.691 0.56 0.796 0.721 0.
_Bert+Muse+Aff_App_ 0.53 0.803 0.723 0.666 0.563 0.793 0.719 0.
_Bert+Muse+Contrast_ 0.556 0.827 0.751 0.692 0.557 0.793 0.718 0.
_Bert+Muse+LIWC_ 0.536 0.8 0.721 0.668 0.577 0.788 0.718 0.
_Bert+Muse+SVerb_ 0.546 0.819 0.741 0.683 0.568 0.79 0.718 0.
_Bert+Muse+W_64_ 0.554 0.792 0.716 0.673 **0.596 0.816 0.748 0.**
_Bert+Muse+W_128_ 0.556 0.791 0.716 0.674 0.529 0.8 0.719 0.
_Bert+Muse+W_All_ **0.582 0.819 0.748 0.701** 0.473 0.825 0.738 0.
_Bert+Muse+LingAll_ 0.517 0.804 0.721 0.661 **0.602 0.817 0.749 0.**
_Bert-baseline_ 0.472 0.803 0.693 0.638 0.472 0.803 0.693 0.
**Table 5**
The impact of the views on MvAttLSTM for irony detection. The ignored view is denoted by (×) symbol whereas
the included views are denoted by (
√
) symbol.
Model Ling Muse Bert _F_ (^1) _iro F_ (^1) _no_ − _iro F_ (^1) _Micro F_ (^1) _Macro
IroSvA’19-es_
×
√ √
0.627 0.835 0.771 0.
_Contextual-self_
√
×
√
√ √ 0.593 0.804 **0.736 0.**
× 0.611 0.819 0.753 0.
×
√ √
0.611 0.827 0.761 0.
_Contextual-multi_
√
×
√
√ √ 0.607 0.788 0.724 0.
× 0.592 0.780 **0.714 0.**
_IroSvA’19-mx_
×
√ √
0.546 0.824 0.746 0.
_Contextual-self_
√
×
√
√ √ 0.511 0.788 **0.704 0.**
× 0.620 0.804 0.741 0.
×
√ √
0.530 0.776 0.696 0.
_Contextual-multi_
√
×
√
√ √ 0.503 0.793 **0.708 0.**
× 0.565 0.800 0.726 0.
_IroSvA’19-cu_
×
√ √
0.557 0.793 **0.718 0.**
_Contextual-self_
√
×
√
√ √ 0.559 0.803 0.728 0.
× 0.576 0.824 0.751 0.
×
√ √
0.528 0.805 **0.724 0.**
_Contextual-multi_
√
×
√
√ √ 0.570 0.786 0.714 0.
× 0.520 0.827 0.746 0.


```
Table 6
Comparison with state-of-the-art methods for irony detection in Spanish variants (IroSvA’19).
IroSvA’19-es IroSvA’19-mx IroSvA’19-cu IroSvA’
```
Ranking Team _F_ (^1) _Macro F_ (^1) _Macro F_ (^1) _Macro F_ (^1) _AVG_
(*) _mxM_ v _AttLSTMmultiContextual_ **0.716 0.732 0.665 0.**
(**) _esM_ v _AttLSTMmultiContextual_ **0.755** 0.674 **0.678 0.**
(***) _cuM_ v _AttLSTMmultiContextual_ 0.710 0.638 **0.709 0.**
1st _ELiRF-UPV_ 0.717 0.680 0.653 0.
2nd _CIMAT_ 0.645 0.671 0.660 0.
3th _JZaragoza_ 0.661 0.67 0.616 0.
4th _ATC_ 0.651 0.645 0.594 0.
... ... ... ... ... ...
14th _UO_ 0.511 0.489 0.499 0.
first positions in the ranking. This fact shows the effectiveness
of our model in addressing the problem of irony detection in
multiple variants of Spanish.
_4.4. Results in satire detection in spanish variants_
Ironyandsatirearebothindirectformsofcommunicationthat
are strongly related to each other. These forms aim at commu-
nicating in implicit ways complex meanings which often aim at
criticizing, offending or hurting a victim. The major differences
between them are based on the intention of the author and
the linguistic resources used to effectively communicate the real
meaning. In this section, we present an evaluation of our model
on two corpora of satirical tweets ( _Salas’17 and Barbieri’15_ ) for
analyzing the feasibility of our model for satire detection in
two Spanish variants (Castilian, and Mexican). Conversely, to the
_IroSvA’19_ corpus, these corpora are not explicitly divided into
train and test, then we use 5-fold cross-validation to compute
the generalization capability of our model in each corpus. In
each iteration 80% of the data was used for training meanwhile
the remainder 20% was considered for testing purpose. Also,
to calibrate the hyperparameters of the model, the training set
was split into two subsets 90% to train the model and 10% for
validation purpose. For each corpus, the hyperparameters were
tuned independently.
The results of our model on _Salas’17_ and _Barbieri’15_ are sum-
marized in Table 7. In this table only the results using the _Early
fusion_ strategy are reported, due to the other fusion method
obtained relatively worse results. At a first glance, in Table 7 can

### be observed that both settings of the model M v AttLSTM

```
Early
self and
```
### M v AttLSTM

_Early
multi_ achieved similar results, even when the model
which uses multi-head attention showed a slight improvement
at the expense of more trainable parameters.
Concretely,fortheCastilianvariantinbothcorpora _Barbieri’_

### and Salas’17-es the model with self-attention M v AttLSTMEarlyself ob-

tained good results when the _Linguistic-view_ is used to inform
themodel.Particularly,appraisalfeatures( _Aff_App_ )wasthemost
relevant for satire detection in _Salas’17-es_ and the second bet-
ter in _Barbieri’15_. Moreover, the features obtained by using the
Wilcoxon test showed a good performance, resulting the second
more relevant _W_64_ and _W_All_ in _Salas’17-es_ , and _W_128_ the

### most relevant in Barbieri’15-es. In the case of M v AttLSTM

_Early
multi_ the
best results for both Castilian corpora were achieved using the
64 best-ranked features _W_64_ according to the Wilcoxon test.
RegardingtheresultsontheMexicanvariant,theyweredifferent
to those achieved on the Castilian tweets. Particularly, the model

### M v AttLSTMmultiEarly perform better when the features related to po-

### larity opposition Contrast were used whereas M v AttLSTMmultiEarly ob-

tained the best performance when psycho-linguistic _LIWC_ fea-
tures were considered. Moreover, it can be observed that _Bert-
baseline_ obtained very competitive results on the three corpora.

```
This fact, confirmed that the representations leaned by the BERT
model are good enough to discriminate between satirical and
non-satirical tweets.
Inaseconddirection,weaimatexploringtheroleofthethree
viewsproposedtoinformMvAttLSTM.Forthat,weevaluatedthe
modelignoringoneviewandincludingtheremainingtwo.Inthis
experiment, the linguistic view ( Ling ) represents the subsets of
features that achieved the best F1-macro (see Table 7). As can
```
be shown in Table 8 the best _F_ (^1) _Macro_ in all corpora was obtained
when all views were used together. In general, we observed
that for the three variants, ignoring _Bert-view_ caused the most
significant drop in performance of both settings of _MvAttLSTM_.
In order to have a comparison with the performance obtained
by other methods proposed in the literature, we compare our
models with the results presented in [54,55]. According to our
knowledge, these works are the only two that addressing the
problemofsatiredetectioninSpanish.InTable9wecompareour
model with three methods ( _SMO+LIWC-ALL_ , _BayesNet+LIWC-ALL_ ,
_J48+LIWC-ALL_ ) proposed in [55] and three methods (SVM+W-B,
SVM+Intrinsic, SVM+ALL) introduced in [54] for satire detection.
The methods proposed in [55] are based on machine learning
combined with hand-crafted features, particularly features de-
rived from LIWC. The major difference among these methods is
the machine learning algorithm used. The methods were evalu-
ated using precision ( _Psat_ ) recall ( _Rsat_ ) and F1 score ( _F_ (^1) _sat_ ) on the
positive class (satirical tweets).
In the same fashion, the methods proposed in [54] differ from
eachotherinthefeaturesemployedtodescribethesatiricaltexts:
_SVM+W-B_ considers features based on word n-grams whereas
_SVM+Intrinsic_ employs linguistic features which are
topic-independent,andfinally _SVM+ALL_ combinesbothsubgroups
offeatures.Inthiscase,themethodswereevaluatedusing _F_ (^1) _Macro_.
Toestablishafaircomparisonwiththepreviousworks,weevalu-
atedtheperformanceofourmodelsthatachievedthebestresults

### on each corpus independently ( salasEsM v AttLSTMmultiEarly ,

### salasMxM v AttLSTMmultiEarly and barbEsMxM v AttLSTMmultiEarly ) and we

```
reevaluated the models on the two remaining corpora. As can be
```
### observedinTable9thethreesettingsofourmodel M v AttLSTM

```
Early
multi
outperformed by almost 10% points the results achieved by the
best methods reported in [54,55].
```
```
4.5. Discriminating between irony and satire
```
```
Some figurative languages devices like irony and satire are
difficult to distinguish from each other due to they share several
characteristics,evencanbenested.Forinstance,satirecanappeal
toironyforcommunicatingindirectandcomplexmeaningsoften
aimingatcensoringorcriticizingpeoples,things,socialandmoral
norms in an ironical way. Furthermore, to evaluate our model
beyond irony vs. non-irony and satire vs. non-satire scenarios we
evaluatedthecapabilityofourmodelfordiscriminatingbetween
both phenomena.
```

```
Table 7
MvAttLSTM for satire detection in Spanish variants Salas’17 and Barbieri’.
```
_Views F_ (^1) _sat F_ (^1) _no_ − _sat F_ (^1) _Micro F_ (^1) _Macro F_ (^1) _sat F_ (^1) _no_ − _sat F_ (^1) _Micro F_ (^1) _Macro
M_ v _AttLSTMselfEarlySalas’17-es M_ v _AttLSTMmultiheadEarly Salas’17-es
Bert+Muse+Aff_All_ 0.958 0.958 0.958 0.958 0.956 0.956 0.956 0.
_Bert+Muse+Aff_Emo_ 0.957 0.958 0.958 0.958 0.959 0.96 0.959 0.
_Bert+Muse+Aff_App_ **0.96 0.961 0.961 0.961** 0.958 0.959 0.958 0.
_Bert+Muse+Contrast_ 0.956 0.958 0.957 0.957 0.959 0.96 0.959 0.
_Bert+Muse+LIWC_ 0.941 0.944 0.943 0.943 **0.959 0.96 0.96 0.**
_Bert+Muse+SVerb_ 0.959 0.959 0.959 0.959 0.959 0.959 0.959 0.
_Bert+Muse+W_64_ **0.959 0.96 0.96 0.96 0.964 0.965 0.964 0.**
_Bert+Muse+W_128_ 0.955 0.956 0.955 0.955 0.953 0.954 0.954 0.
_Bert+Muse+W_All_ **0.959 0.96 0.96 0.96 0.96 0.96 0.96 0.**
_Bert+Muse+LingAll_ 0.957 0.958 0.958 0.958 0.953 0.954 0.954 0.
_Bert-baseline_ 0.938 0.925 0.924 0.924 0.938 0.925 0.924 0.
_M_ v _AttLSTMselfEarlySalas’17-mx M_ v _AttLSTMmultiheadEarly Salas’17-mx
Bert+Muse+Aff-All_ 0.964 0.964 0.964 0.964 0.956 0.955 0.956 0.
_Bert+Muse+Aff-Emo_ 0.948 0.947 0.948 0.948 0.956 0.955 0.956 0.
_Bert+Muse+Aff-App_ 0.947 0.946 0.947 0.947 0.952 0.951 0.952 0.
_Bert+Muse+Contrast_ **0.97 0.969 0.969 0.969** 0.956 0.954 0.955 0.
_Bert+Muse+LIWC_ 0.965 0.964 0.965 0.965 **0.969 0.969 0.969 0.**
_Bert+Muse+SVerb_ **0.967 0.967 0.967 0.967 0.967 0.966 0.966 0.**
_Bert+Muse+W-64_ 0.961 0.96 0.96 0.96 0.96 0.959 0.959 0.
_Bert+Muse+W-128_ **0.967 0.966 0.967 0.967** 0.957 0.957 0.957 0.
_Bert+Muse+W_All_ 0.966 0.966 0.966 0.966 0.964 0.963 0.963 0.
_Bert+Muse+Ling-All_ 0.961 0.960 0.961 0.961 0.966 0.965 0.966 0.
_Bert-baseline_ 0.941 0.950 0.951 0.951 0.941 0.950 0.951 0.
_M_ v _AttLSTMselfEarlyBarbieri’15-es M_ v _AttLSTMmultiheadEarly Barbieri’15-es
Bert+Muse+Aff_All_ 0.953 0.952 0.953 0.953 **0.955 0.954 0.955 0.**
_Bert+Muse+Aff_Emo_ 0.954 0.953 0.954 0.954 0.953 0.952 0.952 0.
_Bert+Muse+Aff_App_ **0.956 0.955 0.955 0.955** 0.952 0.95 0.951 0.
_Bert+Muse+Contrast_ 0.951 0.95 0.951 0.951 0.953 0.952 0.952 0.
_Bert+Muse+LIWC_ 0.954 0.953 0.954 0.954 0.954 0.953 0.954 0.
_Bert+Muse+SVerb_ 0.948 0.948 0.948 0.948 0.954 0.954 0.954 0.
_Bert+Muse+W_64_ 0.954 0.954 0.954 0.954 **0.958 0.957 0.957 0.**
_Bert+Muse+W_128_ **0.956 0.956 0.956 0.956** 0.954 0.952 0.953 0.
_Bert+Muse+W_All_ 0.95 0.949 0.949 0.949 0.948 0.946 0.947 0.
_Bert+Muse+LingAll_ 0.953 0.952 0.952 0.952 0.954 0.954 0.954 0.
_Bert-baseline_ 0.942 0.947 0.947 0.947 0.942 0.947 0.947 0.
**Table 8**
The impact of the views on MvAttLSTM for satire detection. The ignored view is denoted by (×) symbol whereas
the included views are denoted by (
√
) symbol.
_Model_ Ling Muse Bert _F_ (^1) _sat F_ (^1) _no_ − _sat F_ (^1) _Micro F_ (^1) _Macro
Salas’17-es_
×
√ √
0.958 0.959 0.959 0.
_Early-self_
√
×
√
√ √ 0.955 0.956 0.955 0.
× 0.710 0.720 **0.721 0.**
×
√ √
0.952 0.954 0.953 0.
_Early-multi_
√
×
√
√ √ 0.952 0.954 0.953 0.
× 0.603 0.685 **0.654 0.**
_Salas’17-mx_
×
√ √
0.958 0.959 0.959 0.
_Early-self_
√
×
√
√ √ 0.958 0.957 0.958 0.
× 0.719 0.858 **0.822 0.**
×
√ √
0.956 0.956 0.956 0.
_Early-multi_
√
×
√
√ √ 0.936 0.931 0.934 0.
× 0.649 0.683 **0.681 0.**
_Barbieri’15-es_
×
√ √
0.95 0.949 0.95 0.
_Early-self_
√
×
√
√ √ 0.952 0.950 0.951 0.
× 0.743 0.728 **0.736 0.**
×
√ √
0.939 0.935 0.937 0.
_Early-multi_
√
×
√
√ √ 0.951 0.950 0.951 0.
× 0.743 0.702 **0.725 0.**
In this sense, the first step was building the satire-irony cor-
pus. For that purpose, we merged the 1000 ironic tweets of the
_IroSvA’19_ corpus with the 2500 satirical tweets of _Salas’17_ for
the Mexican and Castilian variants of the Spanish independently.
After that, we reevaluated the best model for irony detection

### ( iroM v AttLSTMmultiContextual ) and the best model for satire detection

### ( satM v AttLSTMmultiEarly ) in each variant (see Section 4.4). In Table 10,

```
we present the results obtained. As can be observed, the results
show that our model is able to effectively discriminate satire
from irony in both variants (es and mx) with an effectiveness
```
_F_ (^1) _Macro_ = 0 .987 for Castilian tweets and _F_ (^1) _Macro_ = 0 .978 for


```
Table 9
Comparison with state-of-the-art methods for satire detection in Spanish variants.
Salas’17-mx Salas’17-es Barbieri’15-es
```
_Method Psat Rsat F_ (^1) _sat Psat Rsat F_ (^1) _sat F_ (^1) _Macro
SMO+LIWC-ALL_ 0.855 0.855 0.855 0.846 0.84 0.84 –
_BayesNet+LIWC-ALL_ 0.757 0.756 0.756 0.734 0.734 0.734 –
_J48+LIWC-ALL_ 0.752 0.752 0.752 0.774 0.774 0.774 –
_SVM+W-B_ – – – – – – 0.
_SVM+Intrinsic_ – – – – – – 0.
_SVM+ALL_ – – – – – – 0.
_salasEsM_ v _AttLSTMmultiEarly_ 0.966 0.973 **0.969** 0.969 0.950 **0.959 0.**
_salasMxM_ v _AttLSTMmultiEarly_ 0.951 0.969 **0.960** 0.973 0.955 **0.964 0.**
_barbEsM_ v _AttLSTMmultiEarly_ 0.951 0.969 **0.960** 0.973 0.955 **0.964 0.
Table 10**
MvAttLSTM for irony vs. satire detection in Castilian and Mexican tweets.
_Views F_ (^1) _iro F_ (^1) _sat F_ (^1) _Micro F_ (^1) _Macro F_ (^1) _iro F_ (^1) _sat F_ (^1) _Micro F_ (^1) _Macro
sat_ − _M_ v _AttLSTMmultiheadEarly es sat_ − _M_ v _AttLSTMEarlymultiheadmx
Bert+Muse+Aff_All_ 0.950 0.980 0.972 0.965 0.953 0.981 0.973 0.
_Bert+Muse+Aff_Emo_ **0.979 0.992 0.988 0.985 0.968 0.987 0.982 0.**
_Bert+Muse+Aff_App_ **0.981 0.992 0.989 0.987** 0.955 0.982 0.975 0.
_Bert+Muse+Contrast_ 0.957 0.983 0.976 0.970 **0.964 0.986 0.979 0.**
_Bert+Muse+LIWC_ 0.976 0.990 0.986 0.983 0.952 0.981 0.972 0.
_Bert+Muse+SVerb_ 0.886 0.957 0.938 0.921 0.948 0.980 0.971 0.
_Bert+Muse+W_64_ 0.953 0.982 0.974 0.968 0.948 0.979 0.970 0.
_Bert+Muse+W_128_ 0.953 0.982 0.974 0.967 0.950 0.980 0.971 0.
_Bert+Muse+W_All_ 0.956 0.983 0.975 0.970 0.888 0.963 0.945 0.
_Bert+Muse+LingAll_ 0.920 0.966 0.952 0.943 0.952 0.981 0.973 0.
_Bert-baseline_ 0.983 0.991 0.987 0.984 0.963 0.975 0.964 0.
_iro_ − _M_ v _AttLSTMmultiheadContextuales iro_ − _M_ v _AttLSTMContextualmultiheadmx
Bert+Muse+Aff_All_ **0.966 0.987 0.981 0.976** 0.962 0.985 0.979 0.
_Bert+Muse+Aff_Emo_ 0.944 0.975 0.966 0.959 **0.968 0.987 0.982 0.**
_Bert+Muse+Aff_App_ **0.966 0.986 0.981 0.976** 0.963 0.985 0.979 0.
_Bert+Muse+Contrast_ 0.963 0.986 0.980 0.975 0.963 0.985 0.979 0.
_Bert+Muse+LIWC_ 0.965 0.986 0.980 0.975 0.955 0.982 0.974 0.
_Bert+Muse+SVerb_ 0.963 0.986 0.979 0.974 0.964 0.985 0.979 0.
_Bert+Muse+W_64_ 0.947 0.980 0.971 0.963 0.963 0.985 0.979 0.
_Bert+Muse+W_128_ 0.964 0.986 0.980 0.975 0.964 0.986 0.979 0.
_Bert+Muse+W_All_ 0.929 0.965 0.953 0.947 0.964 0.986 0.980 0.
_Bert+Muse+LingAll_ **0.966 0.987 0.981 0.976** 0.962 0.985 0.979 0.
_Bert-baseline_ 0.983 0.991 0.987 0.984 0.963 0.975 0.964 0.
Mexican tweets. Also, _Bert-baseline_ showed very high results on
both corpora.

### Concretely, the model satM v AttLSTM

_Early
multi_ achieves, in general,
thebestperformanceinbothvariants.Alltheseresultsmakeevi-
dentthatthoseviewssuchas _Linguistic-view_ and _Muse-views_ have
a low impact on the model. A possible reason is that these views
werelearnedwithoutanysupervisionrelatedtothespecifictask.
Conversely, _Bert-view_ , which is a task-dependent view, has a
major impact on the model effectiveness, particularly due to this
view was learned in a supervised way and it is strongly related
to the specific task dataset. Regarding the linguistics views, the

### best results of satM v AttLSTM

_Early
multi_ in the Mexican and Castilian
variants were _Aff_Emo_ and _Aff_App_ respectively. According to
theseresults,wecouldappreciatethataffectiveinformationwas,
in general, the most relevant to inform our model for capturing
useful information to detect irony and satire in Spanish variants.
Table 11 shows the impact of the views to inform our model.
The obtained results are aligned with the results achieved in
the task of satire detection. As can be observed, ignoring _Bert-
view_ caused the most significant drop in the performance of the

### M v AttLSTM modelwhereasthemodelislesssensitivetoexclude

_Ling-view_ and _Muse-view_.
Analyzing the results presented in Tables 7 and 10, we could
appreciate that our model is better discriminating irony from
satire than irony from no-irony and satire from no-satire. This
behavior is caused by the nature of the dataset. Satirical tweets
were retrieved from different topics than ironic tweets. This fact

```
introducesabiaswithrespecttothetopicsdiscussedintheironic
and satirical tweets.
```
```
4.6. Validating the robustness of the models in humor recognition
```
```
Our final experiment aims at investigating the robustness of
our model for recognizing humorous tweets written in Spanish.
We are intrigued by the fact that irony and satire are two phe-
nomena that are strongly related to humor. Particularly, some
theoretical works comments about the relation between humor-
irony [167,168] and humor-satire [19].
Inthissense,weevaluatedourmodelwiththecorpus HAHA’
and the results are shown in Table 12. In this case, only the
results achieved by our model using the Early fusion strategy are
presented due to the Contextual fusion method obtained worse
results. At a first glance, we can appreciate that our model
achieves very similar results for both attention mechanisms, al-
```
### thoughthemodel M v AttLSTM

```
Early
multi showsaslightimprovementin
```
terms of _F_ (^1) _humor_. Another important aspect to notice is regarding

### the linguistic views, particularly the model M v AttLSTM

```
Early
self that
performs better when affective features such as Aff_Emo and
```
### Aff_App are used. However, the model M v AttLSTM

```
Early
multi obtains its
bestresultswhenmorefeaturesareconsidered,particularlythose
best-ranked according to the Wilcoxon test W_128 and W_All.
This behavior is aligned with the results presented in [145]. Also,
in this corpus Bert-baseline achieved competitive results in com-
parison with our proposed models. With respect to the impact of
```

```
Table 11
The impact of the views on MvAttLSTM for irony and satire distinguishing. The ignored view is
denoted by (×) symbol whereas the included views are denoted by (
√
) symbol.
```
_Model_ Ling Muse Bert _F_ (^1) _iro F_ (^1) _sat F_ (^1) _Micro F_ (^1) _Macro
Castilian variant_
×
√ √
0.957 0.983 0.976 0.
_Early-multi (sat)_
√
×
√
√ √ 0.966 0.987 0.981 0.
× 0.942 0.977 **0.967 0.**
×
√ √
0.961 0.985 0.978 0.
_Contextual-multi(iro)_
√
×
√
√ √ 0.967 0.987 0.981 0.
× 0.873 0.930 **0.911 0.**
_Mexican variant_
×
√ √
0.951 0.981 0.973 0.
_Early-multi(sat)_
√
×
√
√ √ 0.948 0.979 0.970 0.
× 0.407 0.896 **0.824 0.**
×
√ √
0.964 0.986 0.979 0.
_Contextual-multi(iro)_
√
×
√
√ √ 0.966 0.986 0.981 0.
× 0.532 0.730 **0.740 0.
Table 12**
MvAttLSTM for humor recognition in Spanish tweets ( _HAHA’19_ )
_Views F_ (^1) _hum F_ (^1) _no_ − _hum F_ (^1) _Micro F_ (^1) _Macro F_ (^1) _hum F_ (^1) _no_ − _hum F_ (^1) _Micro F_ (^1) _Macro
M_ v _AttLSTMselfEarly M_ v _AttLSTMmultiheadEarly
Bert+Muse+Aff_All_ 0.802 0.876 0.848 0.839 0.794 0.879 0.848 0.
_Bert+Muse+Aff_Emo_ **0.804 0.881 0.852 0.842** 0.799 0.877 0.847 0.
_Bert+Muse+Aff_App_ **0.804 0.878 0.85 0.841** 0.798 0.879 0.849 0.
_Bert+Muse+LIWC_ 0.796 0.881 0.85 0.839 0.798 0.88 0.849 0.
_Bert+Muse+SVerb_ 0.798 0.88 0.849 0.839 0.797 0.877 0.847 0.
_Bert+Muse+W_64_ 0.803 0.881 0.852 0.842 0.801 0.879 0.85 0.
_Bert+Muse+W_128_ 0.798 0.881 0.85 0.839 **0.806 0.879 0.851 0.**
_Bert+Muse+W_All_ 0.795 0.88 0.849 0.838 **0.804 0.882 0.853 0.**
_Bert+Muse+LingAll_ 0.802 0.872 0.845 0.837 0.796 0.88 0.849 0.
_Bert-baseline_ 0.802 0.864 0.84 0.833 0.802 0.864 0.84 0.
**Table 13**
The impact of the views on MvAttLSTM for humor recognition. The ignored view is denoted by (×) symbol whereas
the included views are denoted by (
√
) symbol.
_Model_ Ling Muse Bert _F_ (^1) _hum F_ (^1) _no_ − _hum F_ (^1) _Micro F_ (^1) _Macro
HAHA’_
×
√ √
0.792 0.88 0.848 0.
_Early-self_
√
×
√
√ √ 0.79 0.874 0.843 0.
× 0.783 0.876 **0.842 0.**
×
√ √
0.795 0.88 0.848 0.
_Early-multi_
√
×
√
√ √ 0.784 0.878 0.844 0.
× 0.781 0.881 **0.845 0.**
the views in our models, we observed in Table 13 that _Bert-view_
and _Muse-view_ are more important than _Linguistic-view_.

### We compare the results of M v AttLSTM

_Early
multi_ with those of the
participating systems in the shared task at _HAHA’19_ organized
in the framework of _IberLEF’19_. In this task, the systems were
ranked according to the official measure F1 score in the humor
class, although also and _Acc_ was reported. As can be observed in
Table14,theresultsobtainedbyourmodelareverycompetitive,

obtaining the fourth position of the ranking according to _F_ (^1) _humor_
and the third position in terms of _Acc_ out of 18 systems. The
performance of our model is similar to the best-ranked sys-
tem _Adilism_ in terms of F1. However, the difference in terms
of precision and recall shows that the _Adilism_ system is better
at detecting a major number of humorous tweets whereas our
modelisbetteratdetectingthehumoroustweets.Asfuturework,
a deeper study is required to analyze the low recall achieved by
our model compared to the _Adilism_ system.

**5. Conclusions and future work**

In this work, we have presented MvAttLSTM, a deep learning-
based method for irony and satire detection in Spanish variants.

```
It is based on an Attentive-LSTM model informed with additional
knowledge learned from three distinct perspectives: Linguistic-
view , MUSE-based view , and BERT-based view. We observed that
our model achieved better performance when it is enriched with
the three proposed views. We have evaluated our model on
the corpus IroSvA’19 for irony and on the corpora Salas’17 and
Barbieri’15 for satire detection in Spanish variants. In both tasks,
the model outperforms the state-of-the-art results. Furthermore,
we have evaluated our model on humor recognition using the
corpus HAHA’19 showing a very competitive behavior. Particu-
larly, linguistic information and deep sentence encoding were
more feasible for irony detection whereas BERT views increased
the performance of satire detection and satire vs. irony detection
(RQ1). Interestingly, the results revealed that affective informa-
tionhelpsindetectingironyandsatire.Particularly,thoserelated
toemotions( Aff_Emo )whicharebasedontheresourcesSenticNet
and SEL; and those related to attitude words ( Aff_App ) based on
the LAM lexicon. Experiments also confirmed that both fusion
strategies are feasible. However Contextual fusion achieved better
performance in relative small corpus like IroSvA’19 , whereas the
Early fusion takes advantage of large and self-annotated corpora
```

```
Table 14
Comparison with state of the art systems for humor recognition in Spanish (HAHA’2019).
```
_Ranking Team Phum Rhum F_ (^1) _hum Acc_
1st _Adilism_ 0.791 0.852 0.821 0.
2sd _Kevin & Hiromi_ 0.802 0.831 0.816 0.
3th _Bfarzin_ 0.782 0.839 0.810 0.
** _M_ v _AttLSTMmultiEarly_ **0.819** 0.792 0.806 **0.**
4th _Jamestjw_ 0.793 0.804 0.798 0.
5th _INGEOTEC_ 0.758 0.819 0.788 0.
6th _BLAIR GMU_ 0.745 0.827 0.784 0.
7th _UO_UPV2_ 0.78 0.765 0.773 0.
... _... ... ... ... ..._
18th _Amrita CEN_ 0.478 0.514 0.495 0.
**Table A.**
Description of the linguistic features used in the Linguistic view to inform the proposed models.
Group Feature Description
Stylistics-based
multiLines It takes into account whether the tweet is composed of multiple lines or not (one vs. many
lines).
lenghtW
lengthC
meanLengthW
Three different features are considered; (i) the number of words, (ii) the number of
characters, and (iii) the means of words’ length in the tweet.
isDialog
nDialogMark
Two distinct features are considered; (i) the tweet contains any line that starts with a long
dash (dialog marker), (ii) the number of lines that start with long dashes.
hashtagsFreq
urlsFreq
emojisFreq
These count the number of hashtags, URLs, and emojis in the tweet, respectively.
exclMarkFreq It counts the exclamation marks in the tweet.
wordRep
wordUpper
wordCharRep
wordWithExcl
Four distinct features are considered: (i) the number of words emphasized by word’s
repetition, (ii) the number of words with emphasis by uppercase, (iii) the number of words
emphasized by character flooding, and (iv) the number of words emphasized by continues
exclamation marks.
alliter It captures the occurrence of simple alliteration in the tweet. For that, we considered a
fixed-length sequence of phonetic prefixes with size=3.
quotation It quantifies the phrases enclosed in a double quote.
Q?A It quantifies the question and answer structures in the tweet.
person_pa It quantifies the number of verbs conjugated in the first, second, third persons and the nouns
and adjectives which agree with such verbal conjugations.
tense_tb It quantifies the usage of different verbal tenses in the tweet.
posN
posV
posA
posR
These count the nouns, verbs, adverbs and adjectives in the tweet.
Punctuation It counts the occurrence of dots, commas, semicolons, and question marks in the tweet.
Content-based
Animal
centered-words
It counts the words that occur in a lexicon of animal names.
Toponym
words
It counts the words that occur in a lexicon of country’s names, capital’s names, city’s names
and nationalities.
ObsceneSexual
words
It counts the words that occur in an in-house lexicon of sexual and obscene words.
( _continued on next page_ )
_(RQ2)_. Unexpectedly we found no strong differences in the effec-
tivenessofourmodelwhenself-attentionormulti-headattention
was considered. However, we appreciated that each attention
type attends distinct linguistic features _(RQ3)_. We are aware that
our model has an important limitation which lies in the lack of
explainabilityaboutwhythemodelusedsomefeaturesfromone
setting to another and from one Spanish variant to the others. As
futurework,wewillaimatinvestigatingothermethodsforfusing
the additional knowledge into our model. Moreover, we plan
to carry out a fine-grained analysis on the impact of linguistic
features joined with the information captured by the attention
mechanismforironyandsatireinterpretability.Finally,wearein-
terested in exploring our model in multilingual and cross-lingual
settings.
**Declaration of competing interest**
Theauthorsdeclarethattheyhavenoknowncompetingfinan-
cial interests or personal relationships that could have appeared
to influence the work reported in this paper.
**Acknowledgments**
The work of the first two authors was in the framework of
the research project MISMIS-FAKEnHATE on MISinformation and
MIScommunicationinsocialmedia:FAKEnewsandHATEspeech
(PGC2018-096212-B-C31), funded by Spanish Ministry of Science
and Innovation, and DeepPattern (PROMETEO/2019/121), funded
by the Generalitat Valenciana, Spain.


**Table A.15** ( _continued_ ).
Group Feature Description

```
Semantic-based
```
```
Antonyms It quantifies the pairs of antonyms that occurs in the tweet. This feature is based on the
antonym’ relations provided by WordNet [169], particularly, for the Spanish language we
used the Multilingual Central Repository (MCR) [170].
LexAmbiguity Three different features are considered; (i) the average of the meanings associated with each
word in the tweet, (ii) the number of meanings for the most ambiguous word in the tweet,
(iii) the gap between the value of two previous features.
DomAmbiguity Conversely, to consider the meanings of the words, in these features we consider the number
of domains assigned to the words. Particularly, three distinct features are considered; (i) the
average of domains associated with each word in the tweet, (ii) the greatest number of
domains that a single word has in the tweet, (iii) the gap between the value of the two
previous features. For obtaining the domains of the words we used the WordNet Domainsc
and SUMOdeach separately.
SVerb_classes These features capture distinct semantic frames of the verbs in the tweet based on ADDESE.e
Negation It counts the negation words in the tweet.
```
```
Affective-based
```
```
SSL_polarity
ESL_polarity
CriSol_polarity
LAM11_polarityf
SenticNet_polarity
```
```
These features count positive and negative words in many sentiment resources. Notice that,
for each resource two features are computed. Particularly, we explore four distinct
dictionaries: Spanish Sentiment Lexicon (SSL) [171], Elhuyar Sentiment Lexicon (ESL) [172],
CriSol lexicon [171], and the lexicon LAM11 introduced in [166]. Moreover, the polarity score
associated with the words and concepts in SenticNet was considered.
emojiPol_pos
emojiPol_neg
```
```
The number of positive and negative emoticons and emojis considering the resource
Emoticons Sentiment [173].
LAM11_attitude
eCrisol_attitudeg
```
```
These features count the number of words according to the three distinct attitude categories
(affect, judgment, and appreciation) proposed in [166]. For that, we considered two lexicons,
(i) the LAM11 lexicon introduced in [166] and an extended version of the CriSol lexicon,
where all words were automatically annotated with attitudes (eCrisol) by using the method
proposed in [174].
EmoCat These features count the number of words according to the six basic emotions provided by
the resource SEL [165].
EmoDim These features are based on the four affective dimensions in SenticNet of the Cambria’s
hourglass of emotions model [148,175]: introspection, temper, attitude and sensitivity.h
```
```
Contrast-basedi
```
```
wordPolCont It computes the gap between the most positive and the most negative word in the tweet.
This feature, consider the distance, in terms of tokens, between the words.
emoTextPol-
Cont
```
```
It computes the polarity difference between emoticons and words in the tweet.
```
```
antConsPolCont It considers the polarity contrast between two parts of the tweet when the tweet is split by a
delimiter. In this work we consider as delimiter some adverbs and punctuation marks.
meanPolPhrase It is the mean of the polarities of the words that belong to phrases enclosed by quotes.
polStandDev It is the standard deviation of the polarities of the words that belong to phrases enclosed by
quotes.
prePastPolCont It computes the polarity difference between the parts of the tweet written in present and
past tenses.
skipGPolRate It computes the rate among skip-grams with polarity opposition on the total of candidate
skip-grams. The candidate skip-grams are those composed of two words (nouns, adjectives,
verbs, adverbs) with skip=1. The skip-grams with polarity opposition are those that match
with the patterns positive-negative, positive-neutral, negative-neutral, and vise-versa.
upperTextPol-
Cont
```
```
It computes the polarity difference between capitalized words and the remainder words in
the tweets.
Psycolinguistic-based LIWC_catj These features count the frequency of words in each category provided by the resource
Linguistic Inquiry and Word Countkdictionary [176].
```
a _p_ is parametric to the three persons used in Spanish grammar.
b _t_ is parametric to the various tense in Spanish grammar i.e., present, past, future, etc.
chttp://wndomains.fbk.eu/hierarchy.html.
dhttp://www.adampease.org/OP/.
ehttp://adesse.uvigo.es/data/clases.php.
f _polarity_ is parametric to the type of sentiment, positive and negative.
g _attitude_ is parametric to the type of attitudes affect, judgment, and appreciation.
hIt is worthy to note that for the Spanish language, we used BabelSenticNet [147]. In this extension of SenticNet, the affective dimensions are sensitivity, attention,
aptitude and pleasantness.
iWith the aim of capturing some types of explicit polarity opposition, we used the features proposed in [177]. The Spanish version of SenticNet was used to determine
the polarity contrast between different parts of the text.
j _cat_ is parametric to the 68 categories in the LIWC 2001 Spanish dictionary.
khttp://www.liwc.net.

**Appendix A. Linguistic features**

```
See Table A.15.
```
```
Appendix B. Best MvAttLSTM hyperparameters for each corpus
```
```
See Tables B.16 and B.17.
```

```
Table B.
Hyperparameters for the Contextual MvAttLSTM model on the IroSvA’19 corpora.
Dataset Model Hyperparameters Model Hyperparameters
IroSvA’19-es Contextual
Self
```
```
batch=
att=self
h=
dp=0.
op=adam
lr=0.
```
```
Contextual
Multi
```
```
batch=
att=multihead
h=
dp=0.
op=adam
lr=0.
IroSvA’19-mx Contextual
Self
```
```
batch=
att=self
h=
dp=0.
op=adam
lr=0.
```
```
Contextual
Multi
```
```
batch=
att=multihead
h=
dp=0.
op=adam
lr=0.
IroSvA’19-cu Contextual
Self
```
```
batch=
att=self
h=
dp=0.
op=rmsprop
lr=0.
```
```
Contextual
Multi
```
```
batch=
att=multihead
h=
dp=0.
op=rmsprop
lr=0.
```
```
Table B.
Hyperparameters for the Early MvAttLSTM model on the satire and humor corpora.
Dataset Model Hyperparameters Model Hyperparameters
Salas’17-es Early
Self
```
```
batch=
att=self
h=
dp=0.
op=adam
lr=0.
```
```
Early
Multi
```
```
batch=
att=multihead
h=
dp=0.
op=adam
lr=0.
Salas’17-mx Early
Self
```
```
batch=
att=self
h=
dp=0.
op=adam
lr=0.
```
```
Early
Multi
```
```
batch=
att=multihead
h=
dp=0.
op=rmsprop
lr=0.
Barbieri’15-es Early
Self
```
```
batch=
att=self
h=
dp=0.
op=rmsprop
lr=0.
```
```
Early
Multi
```
```
batch=
att=multihead
h=
dp=0.
op=adam
lr=0.
HAHA’19 Early
Self
```
```
batch=
att=self
h=
dp=0.
op=adam
lr=0.
```
```
Early
Multi
```
```
batch=
att=multihead
h=
dp=0.
op=rmsprop
lr=0.
```