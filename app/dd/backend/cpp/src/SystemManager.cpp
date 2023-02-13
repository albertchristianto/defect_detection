#include "manager.hpp"
#include <algorithm>
#include <cstring>
#include <aio/logger.hpp>
#include <valarray>

#include <aio/async/sub_thread/sthNoQueue.hpp>
#include <FDASFR_aio_utils/datum.hpp>
#include <FDASFR_aio_utils/utils.hpp>
#include <dbConsumer_aio/dbSender.hpp>
#include <matcher_aio/matcher_brute.hpp>
#include "dataSender.hpp"
#include "date/date.h"

#ifdef USE_PROTECTION
#include <software_protection/hasp_api.h>
#include <software_protection/vendor_code.h>
#pragma comment(lib, "../software_protection/hasp_windows_x64_36127.lib")
#endif

//#define LOG_LEVEL AIO_LOGGER_LEVEL_TRACE

//implementations
namespace FDASFR_AIO
{
	Manager::Manager()
	{
		if (!(AIO_LOGGER_INITIALIZE()))
			aio::Logger::Init(true, AIO_LOGGER_LEVEL_TRACE);// Initialize the logger system. IT IS A MUST TO ADD THIS LINE OF CODE!!!!
		AIO_LOGGER_INFO("The FDASFR system is started!");
	}

	Manager::~Manager()
	{
		this->Stop();
		AIO_LOGGER_INFO("The FDASFR system is closed!");
		aio::Logger::Close();// Close the logger system. IT IS A MUST TO ADD THIS LINE OF CODE!!!!
	}

	int Manager::Init(SystemParam& param)
	{
		const std::lock_guard<std::mutex> lock{ mMtx };
		mSystemLife = true;
		mFdasfrEngine = std::make_shared<FDASFR_AIO::Inference::FDASFR>();
		mDatabase = std::make_shared<FDASFR_AIO::database>("./database/face.db");
		if (mDatabase == nullptr)
			return -4;
		mDatabase->Start();
		mDBConsumerEngine = std::make_shared<Consumer::DBConsumer>();

		for (int i = 0; i < param.MaxAPIs; ++i) {//create available channel api
			mAvailApiIds.push_back(i);//set the available id for api

			std::shared_ptr<ImgInferParam> img_infer_data;
			mImgInferDatas.push_back(img_infer_data);
			std::shared_ptr<ImgInfer> img_infer;
			mImgInfers.push_back(img_infer);
			BASE_THREAD_SP the_thread;
			mSenderThreads.push_back(the_thread);
		}

		mDrawBbox = param.DrawBbox;
		mDrawInfo = param.DrawFrameInfo;
		//AIO_LOGGER_INFO("GOING to fdasfr.cpp");

#ifdef USE_PROTECTION
		CheckLicense();
		if (!mValidLicense)
			return -7;
#endif

		int retval = mFdasfrEngine->Init(mHaspData);
		mDBConsumerEngine->Init(mDatabase);
		if (param.UseAutoRestart == 1)
			mSystemRestarterThread = std::make_shared<std::thread>(&Manager::SystemRestarter, this);

		recog_info = mDatabase->getRecogInfo();

		//AIO_LOGGER_TRACE("Done INIT()!");

		return retval;
	}

	void Manager::Restart()
	{
		AIO_LOGGER_TRACE("Attempting to restart the engine!");
		{
			const std::lock_guard<std::mutex> lock{ mMtx };

			for (int i = 0; i < mImgInferDatas.size(); ++i) {
				if (mImgInfers[i] != nullptr)
					mImgInfers[i]->Pause();
			}

			mFdasfrEngine = std::make_shared<FDASFR_AIO::Inference::FDASFR>();
			int retval = mFdasfrEngine->Init(mHaspData);
			mDBConsumerEngine->Init(mDatabase);

			for (auto& infer_data : mImgInferDatas) {
				if (infer_data != nullptr) {
					infer_data->InputQueue = mFdasfrEngine->GetInputQueue();
					infer_data->DBQueue = mDBConsumerEngine->GetDBQueue();
				}
			}

			recog_info = mDatabase->getRecogInfo();
		}
	}

	void Manager::Stop()
	{
		mSystemLife = false;
		if (mSystemRestarterThread != nullptr)
			mSystemRestarterThread->join();

		for (int i = 0; i < mImgInferDatas.size(); ++i)
			DeleteApi(i);

		mFdasfrEngine->Stop();// stop the inference service
		mDBConsumerEngine->Stop();
		AIO_LOGGER_TRACE("Successfully executing stop function of the manager!");
	}

	bool Manager::IsReady()
	{
		return mFdasfrEngine->IsReady() && mDBConsumerEngine->IsReady();
	}

	double Manager::CheckFPS(int what_to_check)
	{
		if (what_to_check == 0)
			return mSendImageCounter.Get();
		else if (what_to_check == 1)
			return mFdasfrEngine->GetFps(1);//get the FPS of the LPD
		else if (what_to_check == 2)
			return mFdasfrEngine->GetFps(2);//get the FPS of the LPD
		else if (what_to_check == 3)
			return mFdasfrEngine->GetFps(3);//get the FPS of the LPD
		else if (what_to_check == 4) {
			double tmp = 0.0;
			for (auto& infer_data : mImgInferDatas) {
				if (infer_data != nullptr)
					tmp += infer_data->FPS.Get();
			}
			return tmp;
		}
		else
			return 0;
	}

	int Manager::AddApi()
	{
		AIO_LOGGER_TRACE("FDASFR inference engine is opening API");
		int id = -1;
		bool res = true;
		{//critical section of the system
			const std::lock_guard<std::mutex> lock{ mMtx };
			if (!mAvailApiIds.empty())
			{//if we still have available id for the API
				id = mAvailApiIds.front();
				mAvailApiIds.pop_front();
			}
		}

		if (id < 0) {
			//std::cout << "Can't open a new channel" << std::endl;
			AIO_LOGGER_WARN("Can't open a new api! Maximum api is {0}", mImgInfers.size());
			return -3;
		}

		try {
			//create the image inference information
			mImgInferDatas[id] = std::make_shared<ImgInferParam>();
			mImgInferDatas[id]->ApiId = id;
			mImgInferDatas[id]->WorkerId = mWorker_id;
			mWorker_id++;
			mImgInferDatas[id]->DrawBbox = mDrawBbox;
			mImgInferDatas[id]->DrawInfo = mDrawInfo;
			mImgInferDatas[id]->InputQueue = mFdasfrEngine->GetInputQueue();
			mImgInferDatas[id]->DBQueue = mDBConsumerEngine->GetDBQueue();

			//mImgInferDatas[id]->sp_fParam = std::make_shared<Matcher::FilterParameters>();
			//mImgInferDatas[id]->sp_fParam->Sim = c_fParam.Sim;
			//mImgInferDatas[id]->sp_fParam->MinEyeDistance = c_fParam.MinimumEyeDistance;
			//mImgInferDatas[id]->sp_fParam->MaxFace = c_fParam.MaxFace;
			///*std::vector<int> values(c_fParam.BlacklistedIds.intPtr,
			//	c_fParam.BlacklistedIds.intPtr + c_fParam.BlacklistedIds.sizeArr);
			//mImgInferDatas[id]->sp_fParam->BlacklistedIds = values;
			//mImgInferDatas[id]->sp_fParam->FromDate = std::string(c_fParam.FromDate);
			//mImgInferDatas[id]->sp_fParam->ToDate = std::string(c_fParam.ToDate);*/
			//mImgInferDatas[id]->sp_fParam->ROIx = c_fParam.ROIx;
			//mImgInferDatas[id]->sp_fParam->ROIy = c_fParam.ROIy;
			//mImgInferDatas[id]->sp_fParam->ROIw = c_fParam.ROIw;
			//mImgInferDatas[id]->sp_fParam->ROIh = c_fParam.ROIh;
			//mImgInferDatas[id]->sp_fParam->save2DbEveryNFrames = c_fParam.Save2DbEveryNFrames;
			//mImgInferDatas[id]->sp_fParam->nTopResults = c_fParam.nTopResults;

			//mFdasfrEngine->FDSetFilterParameters(mImgInferDatas[id]->sp_fParam);
			////mDatabase->UpdateFilteredRecogInfo(mImgInferDatas[id]->sp_fParam);

			recog_info = mDatabase->getRecogInfo();

			mImgInferDatas[id]->sp_Matcher =
				std::make_shared<Matcher::MatMul>(recog_info);

			mImgInfers[id] = std::make_shared<ImgInfer>(mImgInferDatas[id]);
		}
		catch (std::exception& e) {
			res = false;
			AIO_LOGGER_ERROR("AddApi: {0}", e.what());
		}
		if (!res) {
			DeleteApi(id);
			AIO_LOGGER_ERROR("Failed to open a new api! API id: {0}", id);
			return -1;
		}

		AIO_LOGGER_INFO("Successfully opened 1 api! API id: {0}", id);
		return id;
	}

	int Manager::AddApiFuncPtr(int(*hits_func_ptr)(C_Results&), int(*nots_func_ptr)(C_Results&))
	{
		int api_id = AddApi();//create API using class Function
		if (api_id > -1) {
			BASE_THREAD_SP sender_thread =
				std::make_shared<BASE_THREAD>(mThread_id);
			{//create the input reader
				BASE_WORKER_SP send_worker = std::make_shared<FDASFR_AIO::WDataSender<BASE_DATUM,
					BASE_DATUM_SP>>(mImgInferDatas[api_id]->WorkerId, hits_func_ptr, nots_func_ptr, mImgInfers[api_id]);
				BASE_SUBTHREAD_SP send_subthread = std::make_shared<aio::async::SthNoQueue<BASE_DATUM_SP,
					BASE_WORKER_SP>> (mSubThread_id, std::vector<BASE_WORKER_SP>{ send_worker });
				sender_thread->Add(send_subthread);
				sender_thread->Init();//<-- Init the video reader thread
				//update the id
				//mWorker_id++;
				mSubThread_id++;
				mThread_id++;
			}
			{//critical section of the system
				const std::lock_guard<std::mutex> lock{ mMtx };
				//save the sender thread data into the system
				mSenderThreads[api_id] = sender_thread;
			}
			mSenderThreads[api_id]->Start();
			while (!mSenderThreads[api_id]->IsRunning()) {
				std::this_thread::yield();
			}
			AIO_LOGGER_INFO("Successfully starting thread for executing callback function, API Id: {0}",
				api_id);
		}
		return api_id;
	}

	void Manager::DeleteApi(int api_id)
	{
		if (api_id < 0 && api_id >= mImgInferDatas.size())
			return;
		if (mImgInferDatas[api_id] != nullptr) {
			AIO_LOGGER_INFO("Deleting API id: {0}", api_id);
			mImgInfers[api_id]->Stop();
			mSenderThreads[api_id].reset();//stop the Sender thread
			mImgInfers[api_id].reset();
			mImgInferDatas[api_id].reset();
			size_t tmp;
			{//critical section of the system
				const std::lock_guard<std::mutex> lock{ mMtx };
				mAvailApiIds.push_back(api_id);
				tmp = mAvailApiIds.size();
			}
			AIO_LOGGER_TRACE("Available ID: {0}", tmp);
		}
	}

	int Manager::SendImage(int api_id, C_Image& image, C_FilterParameters& c_fParam)
	{
		if (mImgInferDatas.size() == 0) {
			AIO_LOGGER_ERROR("This system doesn't have image inference mode!!!");
			return -2;
		}
		try {
			cv::Mat tmp = FDASFR_AIO::DataBufferToMat(image.Ptr, image.Height,
				image.Width, image.Depth);

			std::shared_ptr<FilterParameters> filter = std::make_shared<FilterParameters>();
			filter->MaxFace = c_fParam.MaxFace;
			filter->MinEyeDistance = c_fParam.MinimumEyeDistance;
			filter->nTopResults = c_fParam.nTopResults;
			filter->ROIh = c_fParam.ROIh;
			filter->ROIw = c_fParam.ROIw;
			filter->ROIx = c_fParam.ROIx;
			filter->ROIy = c_fParam.ROIy;
			if (c_fParam.Save2DbEveryNFrames > 0)
				filter->save2DbEveryNFrames = c_fParam.Save2DbEveryNFrames;
			else
				filter->save2DbEveryNFrames = -1;
			filter->Sim = c_fParam.Sim;

			mImgInfers[api_id]->SendImage(tmp, filter, image.TimeStamp);

			mSendImageCounter.Increment();
			mSendImageCounter.CheckAndUpdate();
		}
		catch (std::exception& e) {
			AIO_LOGGER_ERROR("API Id {0}: {1}", api_id, e.what());
			return -3;
		}

		return 1;
	}

	int Manager::GetResults(int api_id, C_Results& results)
	{
		if (mImgInferDatas.size() == 0) {
			AIO_LOGGER_ERROR("This system doesn't have image inference mode!!!");
			return -2;
		}
		try {
			std::shared_ptr<std::vector<FDASFR_AIO::DataField>> tmp;

			if (mImgInfers[api_id]->GetResult(tmp, results.TimeStamp))
			{
				results.Count = std::min((int)tmp->size(), C_MAX_OBJECTS);
				for (int j = 0; j < results.Count; ++j)
				{//iterate into every bounding boxes
					results.Candidates[j].x = tmp->at(j).bbox.x;
					results.Candidates[j].y = tmp->at(j).bbox.y;
					results.Candidates[j].w = tmp->at(j).bbox.width;
					results.Candidates[j].h = tmp->at(j).bbox.height;
					results.Candidates[j].Similarity = tmp->at(j).minDistance;
					results.Candidates[j].RealFace = tmp->at(j).realFace;
					//std::string emb = mDatabase->mat2str(tmp->at(j).faceEmbedding);
#if _WIN32
					strcpy_s(results.Candidates[j].Name, tmp->at(j).name.c_str());
					strcpy_s(results.Candidates[j].Uuid, tmp->at(j).uuid.c_str());
					strcpy_s(results.Candidates[j].DateTime, tmp->at(j).timeCreated.c_str());
					//strcpy_s(results.Candidates[j].Features, emb.c_str());
#elif __linux__
					strcpy(results.Candidates[j].Name, tmp->at(j).name.c_str());
					strcpy(results.Candidates[j].Uuid, tmp->at(j).uuid.c_str());
					strcpy(results.Candidates[j].DateTime, tmp->at(j).timeCreated.c_str());
					//strcpy(results.Candidates[j].Features, emb.c_str());
#endif
				}
				return 1;
			}
		}
		catch (std::exception& e) {
			AIO_LOGGER_ERROR("API Id {0}: {1}", api_id, e.what());
		}
		return 0;
	}

	int Manager::GetResults(int api_id, C_Image& image, C_Results& results)
	{
		if (mImgInferDatas.size() == 0) {
			AIO_LOGGER_ERROR("This system doesn't have image inference mode!!!");
			return -2;
		}
		try {
			cv::Mat image_tmp;
			std::shared_ptr<std::vector<FDASFR_AIO::DataField>> results_tmp;
			//AIO_LOGGER_INFO("Starting mImgInfers[api_id]->GetResult!");
			if (mImgInfers[api_id]->GetResult(image_tmp, results_tmp, results.TimeStamp))
			{
				image.TimeStamp = results.TimeStamp;
				image.Ptr = FDASFR_AIO::MatToDataBuffer<unsigned char>(image_tmp);
				image.Width = image_tmp.cols;
				image.Height = image_tmp.rows;
				image.Depth = image_tmp.channels();
				//AIO_LOGGER_INFO("success get the image!");
				results.Count = std::min((int)results_tmp->size(), C_MAX_OBJECTS);
				for (int j = 0; j < results.Count; ++j) {//iterate into every bounding boxes
					//AIO_LOGGER_INFO("success get the bbox");
					results.Candidates[j].x = results_tmp->at(j).bbox.x;
					results.Candidates[j].y = results_tmp->at(j).bbox.y;
					results.Candidates[j].w = results_tmp->at(j).bbox.width;
					results.Candidates[j].h = results_tmp->at(j).bbox.height;
					results.Candidates[j].Similarity = results_tmp->at(j).minDistance;
					results.Candidates[j].RealFace = results_tmp->at(j).realFace;
					//std::string emb = mDatabase->mat2str(results_tmp->at(j).faceEmbedding);
#if _WIN32
					strcpy_s(results.Candidates[j].Name, results_tmp->at(j).name.c_str());
					strcpy_s(results.Candidates[j].Uuid, results_tmp->at(j).uuid.c_str());
					strcpy_s(results.Candidates[j].DateTime, results_tmp->at(j).timeCreated.c_str());
					//strcpy_s(results.Candidates[j].Features, emb.c_str());
#elif __linux__
					strcpy(results.Candidates[j].Name, results_tmp->at(j).name.c_str());
					strcpy(results.Candidates[j].Uuid, results_tmp->at(j).uuid.c_str());
					strcpy(results.Candidates[j].DateTime, tmp->at(j).timeCreated.c_str());
					//strcpy(results.Candidates[j].Features, emb.c_str());
#endif
				}
				//AIO_LOGGER_INFO("success get the data!");
				return 1;
			}
		}
		catch (std::exception& e) {
			std::string log_str = e.what();
			AIO_LOGGER_ERROR(log_str);
		}
		return 0;
	}

	int Manager::GetEmbedding(int api_id, C_DBResults& results)
	{
		using namespace date;
		using namespace std::chrono;

		if (mImgInferDatas.size() == 0) {
			AIO_LOGGER_ERROR("This system doesn't have image inference mode!!!");
			return -2;
		}
		try {
			std::shared_ptr<std::vector<FDASFR_AIO::DataField>> tmp;
			unsigned long long tag = 0;
			if (mImgInfers[api_id]->GetEmbedding(tmp, tag))
			{
				results.Count = std::min((int)tmp->size(), C_MAX_OBJECTS);
				for (int j = 0; j < results.Count; ++j) {//iterate into every bounding boxes

					results.Result[j].x = tmp->at(j).bbox.x;
					results.Result[j].y = tmp->at(j).bbox.y;
					results.Result[j].w = tmp->at(j).bbox.width;
					results.Result[j].h = tmp->at(j).bbox.height;
					std::string emb = mDatabase->mat2str(tmp->at(j).faceEmbedding);
					auto timeNow = floor<seconds>(system_clock::now() + 8h);
					std::string time = format("%F %T", timeNow);
#if _WIN32
					strcpy_s(results.Result[j].Name, tmp->at(j).name.c_str());
					strcpy_s(results.Result[j].Uuid, tmp->at(j).uuid.c_str());
					strcpy_s(results.Result[j].DateTime, time.c_str());
					strcpy_s(results.Result[j].Features, emb.c_str());
#elif __linux__
					strcpy(results.Result[j].Name, tmp->at(j).name.c_str());
					strcpy(results.Result[j].DateTime, time.c_str());
					strcpy(results.Result[j].Features, emb.c_str());
#endif
				}
				return 1;
			}
		}
		catch (std::exception& e) {
			AIO_LOGGER_ERROR("API Id {0}: {1}", api_id, e.what());
		}
		return 0;
	}

	int Manager::GetSearchResults(int api_id, std::vector<std::vector<int>>& indexes)
	{
		if (mImgInferDatas.size() == 0) {
			AIO_LOGGER_ERROR("This system doesn't have image inference mode!!!");
			return -2;
		}
		try {
			std::shared_ptr<std::vector<FDASFR_AIO::DataField>> tmp;

			if (mImgInfers[api_id]->GetSearchResult(tmp))
			{
				int count = std::min((int)tmp->size(), 10);
				//AIO_LOGGER_TRACE("COUNT {0}", count);
				for (int j = 0; j < count; ++j) {//iterate into every bounding boxes
					indexes.push_back(tmp->at(j).indexes);
				}
				//AIO_LOGGER_TRACE("EY");
				return 1;
			}
		}
		catch (std::exception& e) {
			AIO_LOGGER_ERROR("API Id {0}: {1}", api_id, e.what());
		}
		return 0;
	}

	int Manager::SearchImage(int api_id, C_Image& input, C_FilterParameters& c_fParam,
		C_DBResults& dbresults)
	{
		try
		{
			SendImage(api_id, input, c_fParam);

			std::vector<std::vector<int>> indexes;
			std::vector<known> knowns;

			int count = 0;
			while (GetSearchResults(api_id, indexes) != 1) {
				/*std::this_thread::sleep_for(std::chrono::milliseconds(100));
				count++;*/
				std::this_thread::yield();
			}

			if (indexes.size() <= 0) {
				//AIO_LOGGER_ERROR("Search(imgPath)@Manager: No Face");
				return -5;
			}
			else {
				//AIO_LOGGER_INFO("index size: {0}", indexes[0].size());
				//for (int& x : indexes[0]) // if you want to add 10 to each element
				//	x += 1;
				//knowns = mDatabase->getSomeKnown(indexes[0]);
				std::vector<std::string> names;
				for (int& x : indexes[0])
					names.push_back(recog_info->labels[x]);

				knowns = mDatabase->getAllKnownInNames(names);
				//AIO_LOGGER_INFO("known size: {0}", knowns.size());
			}

			dbresults.Count = 0;
			for (int i = 0; i < knowns.size(); i++)
			{
				dbresults.Result[dbresults.Count].Id = knowns[i].id;
				cv::Mat img_decode = cv::imdecode(knowns[i].Image, cv::IMREAD_COLOR);
				dbresults.Result[dbresults.Count].CroppedImage.Ptr = MatToDataBuffer<unsigned char>(img_decode);
				dbresults.Result[dbresults.Count].CroppedImage.Height = img_decode.rows;
				dbresults.Result[dbresults.Count].CroppedImage.Width = img_decode.cols;
				dbresults.Result[dbresults.Count].CroppedImage.Depth = img_decode.channels();
#if _WIN32
				strcpy_s(dbresults.Result[dbresults.Count].Name, knowns[i].Name.c_str());
				strcpy_s(dbresults.Result[dbresults.Count].Uuid, knowns[i].UUID.c_str());
				strcpy_s(dbresults.Result[dbresults.Count].DateTime, knowns[i].TimeStamp.c_str());
				strcpy_s(dbresults.Result[dbresults.Count].Features, knowns[i].Feature.c_str());
#elif __linux__
				strcpy(dbresults.Result[dbresults.Count].Name, knowns[i].Name.c_str());
				strcpy(dbresults.Result[dbresults.Count].DateTime, knowns[i].TimeStamp.c_str());
				strcpy(dbresults.Result[dbresults.Count].Features, knowns[i].Feature.c_str());
#endif

				dbresults.Count++;
			}

			return 1;
		}
		catch (std::exception& e) {
			AIO_LOGGER_ERROR("Search Image@Manager: {0}", e.what());
		}
		return -99;
	}

	//int Manager::EditFilterParam(int api_id, C_FilterParameters& c_fParam)
	//{
	//	if (mImgInferDatas.size() == 0) {
	//		AIO_LOGGER_ERROR("This system doesn't have channel inference mode!!!");
	//		return -2;
	//	}
	//	if (mImgInferDatas[api_id] == nullptr) {
	//		AIO_LOGGER_ERROR("Can not find the corresponding channel data for adding the params. API Id: {0}",
	//			api_id);
	//		return -3;
	//	}

	//	try
	//	{
	//		//mImgInferDatas[api_id]->sp_fParam = std::make_shared<Matcher::FilterParameters>();
	//		//mImgInferDatas[api_id]->sp_fParam->Sim = c_fParam.Sim;
	//		//mImgInferDatas[api_id]->sp_fParam->MinEyeDistance = c_fParam.MinimumEyeDistance;
	//		//mImgInferDatas[api_id]->sp_fParam->MaxFace = c_fParam.MaxFace;
	//		///*std::vector<int> values(c_fParam.BlacklistedIds.intPtr,
	//		//	c_fParam.BlacklistedIds.intPtr + c_fParam.BlacklistedIds.sizeArr);
	//		//mImgInferDatas[api_id]->sp_fParam->BlacklistedIds = values;
	//		//mImgInferDatas[api_id]->sp_fParam->FromDate = c_fParam.FromDate;
	//		//mImgInferDatas[api_id]->sp_fParam->ToDate = c_fParam.ToDate;*/
	//		//mImgInferDatas[api_id]->sp_fParam->ROIx = c_fParam.ROIx;
	//		//mImgInferDatas[api_id]->sp_fParam->ROIy = c_fParam.ROIy;
	//		//mImgInferDatas[api_id]->sp_fParam->ROIw = c_fParam.ROIw;
	//		//mImgInferDatas[api_id]->sp_fParam->ROIh = c_fParam.ROIh;
	//		//mImgInferDatas[api_id]->sp_fParam->save2DbEveryNFrames = c_fParam.Save2DbEveryNFrames;
	//		//mImgInferDatas[api_id]->sp_fParam->nTopResults = c_fParam.nTopResults;

	//		//mFdasfrEngine->FDSetFilterParameters(mImgInferDatas[api_id]->sp_fParam);

	//		////mDatabase->UpdateFilteredRecogInfo(mImgInferDatas[api_id]->sp_fParam);

	//		//mImgInferDatas[api_id]->sp_Matcher->AddParams(recog_info, mImgInferDatas[api_id]->sp_fParam);

	//		return 1;
	//	}
	//	catch (std::exception& e) {
	//		std::string log_str = e.what();
	//		AIO_LOGGER_ERROR(log_str);
	//	}
	//	return -99;
	//}

	int Manager::Enroll(int api_id, const char* uuid, const char* name, const char* imgPath,
		C_FilterParameters& filter, int& id)
	{
		try {
			//AIO_LOGGER_TRACE("GETTING THE FILES");
			cv::Mat img = cv::imread(imgPath);
			//AIO_LOGGER_TRACE("GOT THE FILES");
			FDASFR_AIO::C_Image image;
			image.Ptr = img.data;
			image.Width = img.cols;
			image.Height = img.rows;
			image.Depth = img.channels();
			image.TimeStamp = 0;

			C_DBResults packet;
			ExtractEmbedding(api_id, image, filter, packet);

			if (packet.Count <= 0) {
				AIO_LOGGER_ERROR("Enroll(imgPath)@Manager: No Face");
				id = -1;
				return -5;
			}
			else if (packet.Count > 1) {
				AIO_LOGGER_INFO("Enroll(imgPath)@Manager: More than 1 Face, will take the biggest face");
				id = mDatabase->addKnown(uuid, name, img, packet.Result[0].DateTime, packet.Result[0].Features);
			}
			else if (packet.Count == 1) {
				id = mDatabase->addKnown(uuid, name, img, packet.Result[0].DateTime, packet.Result[0].Features);
			}

			for (auto& obj : mImgInferDatas)
			{
				if (obj != nullptr)
					obj->sp_Matcher->UpdateRInfo(recog_info);
			}
			//AIO_LOGGER_INFO("Enroll SUCCESS");
			return 1;
		}
		catch (std::exception& e) {
			AIO_LOGGER_ERROR("Enroll(imgPath)@Manager: {0}", e.what());
		}
		return -99;
	}
	int Manager::Enroll(int api_id, const char* uuid, const char* name, C_Image image,
		C_FilterParameters& filter, int& id)
	{
		try
		{
			cv::Mat img = DataBufferToMat(image.Ptr, image.Height, image.Width, image.Depth);

			C_DBResults packet;
			ExtractEmbedding(api_id, image, filter, packet);

			if (packet.Count <= 0) {
				AIO_LOGGER_ERROR("Enroll(image)@Manager: No Face");
				id = -1;
				return -5;
			}
			else if (packet.Count > 1) {
				AIO_LOGGER_INFO("Enroll(image)@Manager: More than 1 Face, will take the biggest face");
				id = mDatabase->addKnown(uuid, name, img, packet.Result[0].DateTime, packet.Result[0].Features);
			}
			else {
				//AIO_LOGGER_TRACE("Enroll(image)@Manager: Success!");
				id = mDatabase->addKnown(uuid, name, img, packet.Result[0].DateTime, packet.Result[0].Features);
			}

			for (auto& obj : mImgInferDatas)
			{
				if (obj != nullptr)
					obj->sp_Matcher->UpdateRInfo(recog_info);
			}

			return 1;
		}
		catch (std::exception& e) {
			AIO_LOGGER_ERROR("Enroll(image)@Manager: {0}", e.what());
		}
		return -99;
	}
	int Manager::DeEnroll(int api_id, int id)
	{
		try
		{
			mDatabase->removeKnown(id);
			for (auto& obj : mImgInferDatas)
			{
				if (obj != nullptr)
					obj->sp_Matcher->UpdateRInfo(recog_info);
			}
			return 1;
		}
		catch (std::exception& e) {
			AIO_LOGGER_ERROR("DeEnroll@Manager: {0}", e.what());
		}
		return -99;
	}
	int Manager::UpdateUser(int api_id, int id, const char* name, const char* uuid,
		C_Image& image, const char* timestamp, C_FilterParameters& filter)
	{
		try
		{
			cv::Mat temp = DataBufferToMat(image.Ptr, image.Height, image.Width, image.Depth);
			known entry;
			{
				entry.id = id;
				entry.Name = name;
				entry.UUID = uuid;
				entry.Image = mDatabase->img2vectorchar(temp);
				entry.TimeStamp = timestamp;
			}

			C_DBResults packet;
			ExtractEmbedding(api_id, image, filter, packet);
			if (packet.Count <= 0) {
				AIO_LOGGER_ERROR("Enroll(imgPath)@Manager: No Face");
				return -5;
			}
			else if (packet.Count > 1) {
				AIO_LOGGER_INFO("Enroll(imgPath)@Manager: More than 1 Face, take the biggest face");
				entry.Feature = packet.Result[0].Features;
			}
			else {
				entry.Feature = packet.Result[0].Features;
			}

			mDatabase->updateKnown(entry);

			for (auto& obj : mImgInferDatas)
			{
				if (obj != nullptr)
					obj->sp_Matcher->UpdateRInfo(recog_info);
			}

			return 1;
		}
		catch (std::exception& e) {
			AIO_LOGGER_ERROR("UpdateUser@Manager: {0}", e.what());
		}
		return -99;
	}
	int Manager::RemoveHit(int id)
	{
		try
		{
			mDatabase->removeHit(id);

			return 1;
		}
		catch (std::exception& e) {
			AIO_LOGGER_ERROR("RemoveHit@Manager: {0}", e.what());
		}
		return -99;
	}
	int Manager::RemoveNot(int id)
	{
		try
		{
			mDatabase->removeNot(id);

			return 1;
		}
		catch (std::exception& e) {
			AIO_LOGGER_ERROR("RemoveNot@Manager: {0}", e.what());
		}
		return -99;
	}
	int Manager::DeEnroll_Some(int api_id, C_IntArray ids)
	{
		try
		{
			std::vector<int> vec(ids.intPtr, ids.intPtr + ids.sizeArr);
			mDatabase->removeSomeKnown(vec);
			for (auto& obj : mImgInferDatas)
			{
				if (obj != nullptr)
					obj->sp_Matcher->UpdateRInfo(recog_info);
			}
			return 1;
		}
		catch (std::exception& e) {
			AIO_LOGGER_ERROR("DeEnroll_Some@Manager: {0}", e.what());
		}
		return -99;
	}
	int Manager::RemoveSomeHits(C_IntArray ids)
	{
		try
		{
			std::vector<int> vec(ids.intPtr, ids.intPtr + ids.sizeArr);
			mDatabase->removeSomeHits(vec);

			return 1;
		}
		catch (std::exception& e) {
			AIO_LOGGER_ERROR("RemoveSomeHits@Manager: {0}", e.what());
		}
		return -99;
	}
	int Manager::RemoveSomeNots(C_IntArray ids)
	{
		try
		{
			std::vector<int> vec(ids.intPtr, ids.intPtr + ids.sizeArr);
			mDatabase->removeSomeNots(vec);

			return 1;
		}
		catch (std::exception& e) {
			AIO_LOGGER_ERROR("RemoveSomeNots@Manager: {0}", e.what());
		}
		return -99;
	}
	int Manager::DeEnroll_All(int api_id)
	{
		try
		{
			mDatabase->removeAllKnown();
			for (auto& obj : mImgInferDatas)
			{
				if (obj != nullptr)
					obj->sp_Matcher->UpdateRInfo(recog_info);
			}
			return 1;
		}
		catch (std::exception& e) {
			AIO_LOGGER_ERROR("DeEnroll_All@Manager: {0}", e.what());
		}
		return -99;
	}
	int Manager::RemoveAllHits()
	{
		try
		{
			mDatabase->removeAllHits();

			return 1;
		}
		catch (std::exception& e) {
			AIO_LOGGER_ERROR("RemoveAllHits@Manager: {0}", e.what());
		}
		return -99;
	}
	int Manager::RemoveAllNots()
	{
		try
		{
			mDatabase->removeAllNots();

			return 1;
		}
		catch (std::exception& e) {
			AIO_LOGGER_ERROR("RemoveAllNots@Manager: {0}", e.what());
		}
		return -99;
	}

	void Manager::SystemRestarter()
	{
		AIO_LOGGER_TRACE("LPDR Auto Restart is online!");
		while (true)
		{
			if (!this->IsReady())
				this->Restart();
			std::this_thread::sleep_for(std::chrono::seconds(2));
			if (!mSystemLife)
				break;
		}
		AIO_LOGGER_TRACE("LPDR Auto Restart is offline!");
	}
	void Manager::ExtractEmbedding(int api_id, C_Image& image, C_FilterParameters& c_fParam,
		C_DBResults& results)
	{
		SendImage(api_id, image, c_fParam);

		while (GetEmbedding(api_id, results) != 1) {
			//FDASFR_AIO::Info("Here-waiting!");
			std::this_thread::yield();
		}
		//AIO_LOGGER_TRACE("Got Result!");
	}

	C_SystemInfo Manager::GetInfo()
	{
		int avail;
		{//critical section of the system
			const std::lock_guard<std::mutex> lock{ mMtx };
			avail = mAvailApiIds.size();
		}

		std::string issueDate = "2022-03-08 12:33:17";

		C_SystemInfo info;
		info.NumOfAPIs = avail;
		info.MaxEnroll = -1;
		info.MaxMatchSize = -1;
		strcpy_s(info.IssueDate, issueDate.c_str());
	}

#ifdef USE_PROTECTION
	void Manager::CheckLicense() {
		AIO_LOGGER_INFO("HASP: Checking for a valid license");
		hasp_status_t   status;
		hasp_handle_t   handle;

		status = hasp_login(1302, (hasp_vendor_code_t)vendor_code, &handle);//login to hasp
		if (status) {//if the status is not okay
			mValidLicense = false;
			switch (status) {
			case HASP_FEATURE_NOT_FOUND:
				AIO_LOGGER_ERROR("HASP: Login to default feature failed");
				break;

			case HASP_CONTAINER_NOT_FOUND:
				AIO_LOGGER_ERROR("HASP: No sentinel key with vendor code DEMOMA found");
				break;

			case HASP_OLD_DRIVER:
				AIO_LOGGER_ERROR("HASP: Outdated driver version installed");
				break;

			case HASP_NO_DRIVER:
				AIO_LOGGER_ERROR("HASP: Sentinel driver not installed");
				break;

			case HASP_INV_VCODE:
				AIO_LOGGER_ERROR("HASP: Invalid vendor code");
				break;

			default:
				AIO_LOGGER_ERROR("HASP: Login to default feature failed with status " + std::to_string(status));
			}
			return;
		}
		//read the data from license key
		mValidLicense = true;
		mHaspData.resize(48);
		status = hasp_read(handle, HASP_FILEID_RW, 48, mHaspData.size(), mHaspData.data());
		status = hasp_logout(handle);
		//std::string log_str = "";
		//for (int i = 0; i < mHaspData.size(); ++i)
		//	log_str += std::to_string(mHaspData[i]) + ", ";
		//Logging::Info(log_str);
		switch (status) {
		case HASP_STATUS_OK:
			AIO_LOGGER_INFO("HASP: Log out");
			break;

		case HASP_INV_HND:
			AIO_LOGGER_ERROR("HASP: Handle not active");
			break;

		default:
			AIO_LOGGER_ERROR("HASP: Unknown error when trying to log out");
		}
	}
#endif
}