#ifndef CUDA_OPTIMIZER_H_
#define CUDA_OPTIMIZER_H_

#include"executor.h"
#include"cuda/cuda_utils.h"
#include<cudnn.h>
#include<float.h>

class LowerLevelSGDOptimizerGPU: public AbstractLowerLevelOptimizer {
    private:
        double learning_rate_;
        cudnnHandle_t cudnn_;
        cudnnOpTensorDescriptor_t AddDesc;

    public:
        LowerLevelSGDOptimizerGPU(double learning_rate);
        ~LowerLevelSGDOptimizerGPU();
        void optimize_weights(
                Operator * op, 
                DataType * grad,
                DataType * weight_to_update,
                size_t num_elements
                );
        void SetLearningRate(double new_lr){
            learning_rate_ = new_lr;
        }
};
class SGDOptimizerGPU: public AbstractOptimizer {
    private:
        double learning_rate_;
        LowerLevelSGDOptimizerGPU * lower_level_optimizer_;

    public:
        SGDOptimizerGPU(double learning_rate);
        ~SGDOptimizerGPU();
        void optimize_weights(
                const std::vector<Operator*> operators,
                const std::vector<bool> operator_mask
                );
        // the lower-level optimizer classes provided a lower-level 
        // abstraction to provide more flexibility so that more 
        // sophisticated weight updates (e.g., async. update in
        // pipeline parallel) can be implemented
        AbstractLowerLevelOptimizer * get_lower_level_optimizer();
       void SetLearningRate(double new_lr){
            lower_level_optimizer_->SetLearningRate(new_lr);
        }
};


class LowerLevelAdamOptimizerGPU: public AbstractLowerLevelOptimizer {
    private:
        struct OptimizerStateGPU {
            int t; // step 
            DataType * m_t; // biased estimated first moment
            DataType * v_t; // biased estimated second moment
            DataType * m_t_gpu;
            DataType * v_t_gpu;
            double exp_beta1;
            double exp_beta2;
        };

        double learning_rate_;
        double weight_decay_;
        double beta1_, beta2_;
        double epsilon_;
        std::map<Operator*, OptimizerStateGPU> states_;

        void init_state(Operator * op, size_t num_elements);
        void LaunchOptimizeWeights(
                Operator * op, 
                DataType * grad,
                DataType * weight_to_update,
                size_t num_elements);
    public:
        LowerLevelAdamOptimizerGPU(
                double learning_rate = 1e-3, 
                double weight_decay = 0.,
                double beta1 = 0.9,
                double beta2 = 0.999,
                double epsilon = 1e-8);
        ~LowerLevelAdamOptimizerGPU();
        void optimize_weights(
                Operator * op, 
                DataType * grad,
                DataType * weight_to_update,
                size_t num_elements
                );
        void SetLearningRate(double new_lr){
           // printf("mmmmm\n");
            learning_rate_ = new_lr;
        }
        double get_learning_rate() {
            return learning_rate_;
        }
};
class AdamOptimizerGPU: public AbstractOptimizer {
    private:
        LowerLevelAdamOptimizerGPU * lower_level_optimizer_;

    public:
        AdamOptimizerGPU(
                double learning_rate = 1e-3,
                double weight_decay = 0.,
                double beta1 = 0.9,
                double beta2 = 0.999,
                double epsilon = 1e-8
                );
        ~AdamOptimizerGPU();
        void optimize_weights(
                const std::vector<Operator*> operators,
                const std::vector<bool> operator_mask
                );
        // the lower-level optimizer classes provided a lower-level 
        // abstraction to provide more flexibility so that more 
        // sophisticated weight updates (e.g., async. update in
        // pipeline parallel) can be implemented
        AbstractLowerLevelOptimizer * get_lower_level_optimizer();
        void SetLearningRate(double new_lr){
            //printf("chanege lr! %.9f\n", new_lr);
           // printf("chanege lr!\n");
           // printf("chanege lr!\n");
           // printf("chanege lr!\n");
           // printf("chanege lr!\n");
            lower_level_optimizer_->SetLearningRate(new_lr);
        }
        double get_learning_rate() {
            return lower_level_optimizer_->get_learning_rate();
        }
};
class LearningRateScheduler{
    private:
        const double min_learning_rate_;
        double current_learning_rate_;
        const double factor_;
        double best_loss_;
        const double eps_;
        const int patience_;
        int bad_epochs_;
        int last_epoch_;
        int cooldown_counter_;
        const int cooldown_;
       // AdamOptimizerGPU * optimizer_;
    public:
        LearningRateScheduler(double min_lr, double cur_lr, double factor, double eps, int patience, int cooldown):
        min_learning_rate_(min_lr),current_learning_rate_(cur_lr),factor_(factor),eps_(eps),patience_(patience),cooldown_(cooldown)
        {
            best_loss_ = double(FLT_MAX) ;
            bad_epochs_ = 0;
            last_epoch_ = 0;
            cooldown_counter_ = 0;
        };
        double Step(double loss){
            // optimizer_->SetLearningRate(5e-3);
            
            //this->current_learning_rate_ = 5e-3;
            int epoch = this->last_epoch_ + 1;
            this->last_epoch_ = epoch;
            if(loss < this->best_loss_ ){
                this->best_loss_ = loss;
                this->bad_epochs_ = 0;
            }else{
                this->bad_epochs_ += 1;
            }
            if(this->cooldown_counter_ > 0){
                this->cooldown_counter_ --;
                this->bad_epochs_ = 0;
            }
           // printf("1111\n");
            if(this->bad_epochs_ > this->patience_){
              //   printf("2222\n");
                double old_lr = this->current_learning_rate_;
                double new_lr = std::max(old_lr * this->factor_, this->min_learning_rate_);
                if(old_lr - new_lr > this->eps_){
                  //  printf("3333\n");
                  //  printf("%d, %f, %f, %f\n", 1, this->current_learning_rate_, old_lr, new_lr);
                    printf("\nreduce learning rate from %f\n", this->current_learning_rate_);
                    this->current_learning_rate_  = new_lr;
                    printf("\nreduce learning rate to %f\n", this->current_learning_rate_);
                  //  printf("lr scheduler work\n");
                   // optimizer_->SetLearningRate(5e-3);
                   //  printf("lr scheduler work done\n");
                }
                this->cooldown_counter_ = this->cooldown_;
                this->bad_epochs_ = 0;
            }
            return this->current_learning_rate_;
        }  
};
#endif
