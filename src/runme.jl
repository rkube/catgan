#
using Flux
using Flux.Data: DataLoader
using Zygote
using MLDatasets
using Logging
using TensorBoardLogger
using ArgParse
using Printf

using catgan_flux

# Define all parameters
args = Dict("latent_dim" => 128, "num_epochs" => 100, 
            "mdim" => 64, "batch_size" => 100,
            "optimizer" => "ADAM", "lr_D" => 2e-4, "lr_G" => 2e-4);

s = ArgParseSettings()
@add_arg_table s begin
    "--lr_D"
        help = "Learning rate for the discriminator. Default=0.0002"
        arg_type = Float64
        default = 0.0002
    "--lr_G"
        help = "Learning rate for the generator. Default=0.0002"
        arg_type = Float64
        default = 0.0002
    "--latent_dim" 
        help = "Size of the latent dimension. Default=128"
        arg_type = Int
        default = 128
    "--num_epochs" 
        help = "Number of epochs to train for. Default=100"
        arg_type = Int
        default = 100
    "--mdim"
        help = "Number of channels for discriminator. Default=64"
        arg_type = Int
        default = 64
    "--optimizer"
        help = "Optimizer to use. Defaults to ADAM"
        arg_type = String
        default = "ADAM"
    "--batch_size"
        help = "Batch size. Default=100"
        arg_type = Int
        default = 100
    "--activation"
        help = "Activation function to use, α ∈ leakyrelu, elu, celu, relu, etc."
        arg_type = String
        default = "relu"
    "--activation_alpha"
        help = "Optional parameter to activation functions. Default=0.2"
        arg_type = Float64
        default = 0.1
end

args = parse_args(s)
# Print arguments
for (arg, val) in args
    println("     $arg => $val")
end
tb_logger = TBLogger("../logs_scan1/log")
with_logger(tb_logger) do
    @info "hyperparameters" args
end


# Load dataset
x_train, y_train = CIFAR10.traindata();
x_test, y_test = CIFAR10.testdata();

# Convert dataset to Float32 and rescale to range [-1.0; 1.0]
x_train = 2.0f0 * x_train .- 1f0 |> gpu;
x_test = 2.0f0 * x_test .- 1f0 |> gpu;
train_loader = DataLoader((data=x_train, label=y_train), batchsize=args["batch_size"], shuffle=true);

# Define discriminator and generator
D = get_discriminator_v3(args) |> gpu;
G = get_generator_v3(args) |> gpu;

# Instantiate optimizers and extract relevant paramters of D and G
opt_D = getfield(Flux, Symbol(args["optimizer"]))(args["lr_D"]);
opt_G = getfield(Flux, Symbol(args["optimizer"]))(args["lr_G"]);
# Extract the parameters of the discriminator and generator
θ_D = Flux.params(D);
θ_G = Flux.params(G);


lossvec_G = zeros(args["num_epochs"]);
lossvec_D = zeros(args["num_epochs"]);

for epoch ∈ 1:args["num_epochs"]
    iter = 1
    for (x, y) ∈ train_loader
        # Train the discriminator
        loss_D, back_D = Zygote.pullback(θ_D) do
            # Sample noise and generate a batch of fake data
            y_real = D(x)
            z = randn(Float32, 1, 1, args["latent_dim"], args["batch_size"]) |> gpu;
            y_fake = D(G(z))
            loss_D = -H_of_p(y_real) + E_of_H_of_p(y_real) - E_of_H_of_p(y_fake)
        end
        # Implement literal transcription of Eq.(7). Then do gradient ascent, i.e. minimize 
        # -L(x, θ) by seeding the gradients with -1.0 instead of 1.0
        grads_D = back_D(one(loss_D))
        Flux.update!(opt_D, θ_D, grads_D)
        lossvec_D[epoch] += loss_D / length(train_loader)

        # Train the generator
        loss_G, back_G = Zygote.pullback(θ_G) do
            z = randn(Float32, 1, 1, args["latent_dim"], args["batch_size"]) |> gpu;
            y_fake = D(G(z));
            loss_G = -H_of_p(y_fake) + E_of_H_of_p(y_fake)
        end
        grads_G = back_G(one(loss_G));
        Flux.update!(opt_G, θ_G, grads_G)
        lossvec_G[epoch] += loss_G / length(train_loader)

        if iter % 50 == 0
            z = randn(Float32, 1, 1, args["latent_dim"], args["batch_size"]) |> gpu;
            y_fake = D(G(z))
            y_real = D(x)
            iter % 50 == 0 && @printf "Iter %03d[%03d]: H(y_real) = %8.6f E[H(p_real)] = %8.6f H(y_fake) = %8.6f E[H(p_fake)] = %8.6f\n" iter epoch H_of_p(y_real) E_of_H_of_p(y_real) H_of_p(y_fake) E_of_H_of_p(y_fake)
        end

        iter += 1
    end
    @printf "Epoch [%03d] loss_G=%8.6f loss_D=%8.6f\n" epoch lossvec_G[epoch] lossvec_D[epoch]

    # Output and logging below
    (x, y) = first(train_loader)
    y_real = D(x)
    z = randn(Float32, 1, 1, args["latent_dim"], args["batch_size"]) |> gpu;
    x_fake = G(z);
    y_fake = D(x_fake)
    x_fake = x_fake |> cpu;

    img_fname = @sprintf "G_epoch_%03d.png" epoch
    img_array = save_images(x_fake, img_fname)


    with_logger(tb_logger) do
        @info "performance" H_y_real=H_of_p(y_real) E_H_real=E_of_H_of_p(y_real) H_y_fake=H_of_p(y_fake) E_H_fake=E_of_H_of_p(y_fake) log_step_increment=0
        @info "Losses" loss_G=lossvec_G[epoch] loss_D=lossvec_D[epoch] log_step_increment=0
        @info "Output" gan_output=img_array
    end
end


