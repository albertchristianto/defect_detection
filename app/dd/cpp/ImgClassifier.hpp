#include <nf/inference_core/i_InferEngine.hpp>

namespace dd {
    /// This interface class is used to control a inference engine.
    /// The user of this framework must inherit their deep learning engine implementtation from this class.
    template<typename SpTDatum>
    class ImageClassifier: nf::I_InferEngine<SpTDatum> {
    public:
        /**
         * Class destructor function
         */
        ~ImageClassifier() {}
        /**
         * Function to initialize the inference engine. the user of this framework must implement this functions.
         * @return true if the initialization process is successful
         * @return false if the initialization process is failed
         */
        bool Init() {}
        /**
         * Function to do the inference. the user of this framework must implement this functions.
         * @param the_datas batch data which to be processed
         */
        void Forward(std::vector<SpTDatum> &the_datas) {};
        /**
         * Function to check the readiness of the inference engine. the user of this framework must implement this functions.
         * @return true if the inference engine is ready
         * @return false if the inference engine is not ready
         */
        bool IsReady();
    };
}