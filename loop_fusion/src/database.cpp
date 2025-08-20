#include "database.h"
#include <climits>

DatabaseInterface::DatabaseInterface()
{
}

DatabaseInterface::DatabaseInterface(const std::string voc_path, int DETECTOR)
{
  if (DETECTOR == 0)
  {
    LoadSppVocabulary(voc_path);
  }
  else if (DETECTOR == 1)
  {
    LoadOrbVocabulary(voc_path);
    // BriefVocabularyPtr voc = std::make_shared<BriefVocabulary>();
    // voc->load(voc_path);
    // orb_voc = voc;
    // orb_db->setVocabulary(*orb_voc, false, 0);
    // std::cout << "voc" << orb_voc << std::endl;
  }
}

void DatabaseInterface::LoadSppVocabulary(const std::string voc_path)
{
  SuperpointVocabulary voc_load;
  std::ifstream ifs(voc_path, std::ios::binary);
  boost::archive::binary_iarchive ia(ifs);
  ia >> voc_load; // serialize file to voc
  spp_voc = std::make_shared<SuperpointVocabulary>(voc_load);
  spp_db = std::make_shared<SuperpointDatabase>();
  spp_db->setVocabulary(*spp_voc, false, 0);
  std::cout << "voc" << *spp_voc << std::endl;
  // if(_inverted_file.empty()){
  //   _inverted_file.resize(spp_voc->size());
  // }
}

void DatabaseInterface::LoadOrbVocabulary(const std::string voc_path)
{
  // BriefVocabulary voc_load;
  // std::ifstream ifs(voc_path);
  // boost::archive::binary_iarchive ia(ifs);
  // ia >> voc_load; // serialize file to voc

  // BriefVocabulary voc_load;
  // std::ifstream ifs(voc_path, std::ios::binary);
  // boost::archive::binary_iarchive ia(ifs);
  // ia >> voc_load;
  // serialize
  //  orb_voc = voc;

  orb_voc = std::make_shared<OrbVocabulary>();
  orb_voc->loadFromBinFile(voc_path);
  orb_db = std::make_shared<OrbDatabase>();
  orb_db->setVocabulary(*orb_voc, false, 0);
  std::cout << "voc" << *orb_voc << std::endl;
  //  _inverted_file.resize(spp_voc->size());
}

// void DatabaseInterface::FrameToBow(FramePtr frame, DBoW2::WordIdToFeatures& word_features, DBoW2::BowVector& bow_vector){
//   const Eigen::Matrix<float, 259, Eigen::Dynamic>& features_eigen = frame->GetAllFeatures();
//   FrameToBow(features_eigen, word_features, bow_vector);
// }

// void DatabaseInterface::FrameToBow(const Eigen::Matrix<float, 259, Eigen::Dynamic>& features_eigen,
//     DBoW2::WordIdToFeatures& word_features, DBoW2::BowVector& bow_vector){
//   int N = features_eigen.cols();
//   std::vector<Eigen::Matrix<float, 256, 1>>> features;
//   features.reserve(N);
//   for(int i = 0; i < N; i++){
//     features.emplace_back(features_eigen.block(3, i, 256, 1));
//   }
//   spp_voc->transform(features, bow_vector, word_features);
// }

// void DatabaseInterface::FrameToBow(FramePtr frame, DBoW2::WordIdToFeatures& word_features, DBoW2::BowVector& bow_vector,
//     std::vector<DBoW2::WordId>& word_of_features){
//   const Eigen::Matrix<float, 259, Eigen::Dynamic>& features_eigen = frame->GetAllFeatures();
//   FrameToBow(features_eigen, word_features, bow_vector, word_of_features);
// }

// void DatabaseInterface::addFeatures(const FeatureData &fd_in)
// {
//   Eigen::Matrix<float, 256, Eigen::Dynamic> features;
//   int N = fd_in.size();
//   if (N == 0)
//     return;

//   for (int i = 0; i < N; i++)
//   {
//     // Eigen::Matrix<float, 256, 1> spp_descriptor = features[i].spp_feature.tail(256);
//     features.col(i) = fd_in[i].spp_feature.tail(256);
//   }
//   this->add(features);
// }

void DatabaseInterface::FrameToBow(const FeatureData &features,
                                   DBoW2::WordIdToFeatures &word_features, DBoW2::BowVector &bow_vector, std::vector<DBoW2::WordId> &word_of_features)
{

  // const Eigen::Matrix<float, 259, Eigen::Dynamic>& features_eigen;
  int N = features.size();
  if (N == 0)
    return;

  // normalize
  DBoW2::LNorm norm;
  bool must = spp_voc->m_scoring_object->mustNormalize(norm);
  for (int i = 0; i < N; i++)
  {
    DBoW2::WordId id;
    DBoW2::WordValue w; // w is the idf value if TF_IDF, 1 if TF
    Eigen::Matrix<float, 256, 1> spp_descriptor = features[i].spp_feature.tail(256);
    spp_voc->transform(spp_descriptor, id, w);
    if (w > 0)
    {
      bow_vector.addWeight(id, w);
      word_features[id].emplace_back(i);
      word_of_features.push_back(id);
    }
    else
    {
      word_of_features.push_back(UINT_MAX);
    }
  }

  if (bow_vector.empty())
    return;

  if (must)
  {
    bow_vector.normalize(norm);
  }
  else
  {
    const double nd = bow_vector.size();
    for (DBoW2::BowVector::iterator vit = bow_vector.begin(); vit != bow_vector.end(); vit++)
    {
      vit->second /= nd;
    }
  }
}

// void DatabaseInterface::AddFrame(FramePtr frame){
//   DBoW2::WordIdToFeatures word_features;
//   DBoW2::BowVector bow_vector;
//   FrameToBow(frame, word_features, bow_vector);
//   AddFrame(frame, word_features, bow_vector);
// }

// void DatabaseInterface::AddFrame(const DBoW2::WordIdToFeatures &word_features, const DBoW2::BowVector &bow_vector)
// {
//   // _frame_bow_vectors[frame] = bow_vector;

//   // update inverted file
//   for (auto &kv : word_features)
//   {
//     const DBoW2::WordId &word_id = kv.first;
//     _inverted_file[word_id][frame] = kv.second;
//   }
// }

// void DatabaseInterface::Query(const DBoW2::BowVector& bow_vector, std::map<FramePtr, int>& frame_sharing_words){
//   DBoW2::BowVector::const_iterator vit;
//   for(vit = bow_vector.begin(); vit != bow_vector.end(); ++vit){
//     const DBoW2::WordId word_id = vit->first;
//     for(const auto& kv : _inverted_file[word_id]){
//       FramePtr f = kv.first;
//       if(frame_sharing_words.find(f) == frame_sharing_words.end()){
//         frame_sharing_words[f] = 0;
//       }
//       frame_sharing_words[f]++;
//     }
//   }
// }

double DatabaseInterface::Score(const DBoW2::BowVector &bow_vector1, const DBoW2::BowVector &bow_vector2)
{
  return spp_voc->score(bow_vector1, bow_vector2);
}