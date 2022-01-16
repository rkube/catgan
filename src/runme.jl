#
using Flux
using Flux.Data: DataLoader
using Zygote
using MLDatasets
using Printf

using catgan_flux

# Define all parameters
args = Dict("noise_dim" => 128, "num_epochs" => 100, 
            "mdim" => 64, "batch_size" => 100,
            "optimizer" => "RMSProp", "lr_D" => 2e-4, "lr_G" => 2e-4);

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
            z = randn(Float32, 1, 1, args["noise_dim"], args["batch_size"]) |> gpu;
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
            z = randn(Float32, 1, 1, args["noise_dim"], args["batch_size"]) |> gpu;
            y_fake = D(G(z));
            loss_G = -H_of_p(y_fake) + E_of_H_of_p(y_fake)
        end
        grads_G = back_G(one(loss_G));
        Flux.update!(opt_G, θ_G, grads_G)
        lossvec_G[epoch] += loss_G / length(train_loader)

        if iter % 50 == 0
            z = randn(Float32, 1, 1, args["noise_dim"], args["batch_size"]) |> gpu;
            y_fake = D(G(z))
            y_real = D(x)
            iter % 50 == 0 && @printf "Iter %03d[%03d]: H(y_real) = %8.6f  H(y_fake) = %8.6f\n" iter epoch H_of_p(y_real) H_of_p(y_fake)
        end

        iter += 1
    end
    @printf "Epoch [%03d] loss_G=%8.6f loss_D=%8.6f\n" epoch lossvec_G[epoch] lossvec_D[epoch]
    z = randn(Float32, 1, 1, args["noise_dim"], args["batch_size"]) |> gpu;
    x_fake = G(z) |> cpu;
    img_fname = @sprintf "G_epoch_%03d.png" epoch
    save_images(x_fake, img_fname)
end


