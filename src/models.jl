

export get_discriminator, get_discriminator_v2, get_discriminator_v3, get_generator, get_generator_v2, get_generator_v3

leakyrelu02 = Base.Fix2(leakyrelu, 0.2)

function get_discriminator(args)
    mdim = args["mdim"]
    # Cifar images start out as (32)x(32)x(x)
    return Chain(Conv((4, 4), 3 => mdim, stride=(2, 2), pad=1, bias=false),
                 # Image is now (16)x(16)x(8)
                 BatchNorm(mdim, elu),
                 # 2. block
                 Conv((4, 4), mdim => 2 * mdim, stride=(2, 2), pad=1, bias=false),
                 # Image is now (8)x(8)x(16)
                 BatchNorm(2 * mdim, elu),
                 # 3. block
                 Conv((4, 4), 2 * mdim => 4 * mdim, stride=(2, 2), pad=1, bias=false),
                 # Image is now (4)x(4)x(32)
                 BatchNorm(4 * mdim, elu),
                 # 4. block
                 Conv((4, 4), 4 * mdim => 4 * mdim, bias=false),
                 # Image is now (1)x(1)x(32)
                 BatchNorm(4 * mdim, elu),
                 x -> flatten(x),
                 # state is now (32)x(batch_size)
                 Dense(4 * mdim, 10),
                 x -> softmax(x))

end


function get_discriminator_v2(args)
    D2 = Chain(Conv((3, 3), 3 => 96, elu),
               Conv((3, 3), 96=>96, elu),
               Conv((3, 3), 96=>96, elu),
               MaxPool((2, 2), stride=2),
               BatchNorm(96, elu),
               Conv((3, 3), 96=>192, elu),
               Conv((3, 3), 192=>192, elu),
               Conv((3, 3), 192=>192, elu),
               MaxPool((3, 3), stride=2),
               BatchNorm(192, elu),
               Conv((3, 3), 192=>192, elu),
               Conv((1, 1), 192=>192, elu),
               Conv((1, 1), 192=>10),
               GlobalMeanPool(), x -> elu.(x),
               flatten, softmax)
end



function get_generator(args)
    mdim = args["mdim"] # Model dimension
    latent_dim = args["latent_dim"]   # Latent dimension

    # The input to the generator is a 1x1 image with latent_dim channels
    return Chain(ConvTranspose((4, 4), latent_dim => 4 * mdim, bias=false),
                 # Now imega is (4)x(4)x(4*mdim)
                 BatchNorm(4 * mdim, elu),
                 ConvTranspose((4, 4), 4 * mdim => 2 * mdim, stride=(2, 2), pad=1, bias=false),
                 # Now image is (8)x(8)x(2*mdim)
                 BatchNorm(2 * mdim, elu),
                 ConvTranspose((4, 4), 2 * mdim => mdim, stride=(2, 2), pad=1, bias=false),
                 # Now image is (16)x(16)x(mdim)
                 BatchNorm(mdim, elu),
                 ConvTranspose((4, 4), mdim => 3, tanh, stride=(2, 2), pad=1, bias=false))
                 # Now image is (32)x(32)x(3)
end


function get_generator_v2(args)
    latent_dim = args["latent_dim"]

    return Chain(flatten, Dense(args["latent_dim"], 8*8*192, elu),
                 x -> reshape(x, (8, 8, 192, 100)),
                 ConvTranspose((2, 2), 192=>192, elu, stride=(2, 2)),
                 Conv((5, 5), 192 => 96, elu, pad=SamePad()),
                 Conv((5, 5), 96 => 96, elu, pad=SamePad()),
                 ConvTranspose((2, 2), 96 => 96, stride=(2, 2)),
                 Conv((5, 5), 96 => 96, elu, pad=SamePad()),
                 Conv((5, 5), 96 => 3, elu, pad=SamePad()));

end


# Trying architectures described here:
# https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/

function get_discriminator_v3(args)
    if args["activation"] in ["celu", "elu", "leakyrelu", "trelu"]
        # Now continue: We want to use Base.Fix2
        act = Fix2(getfield(NNlib, Symbol(args["activation"])), Float32(args["activation_alpha"]))
    else
        act = getfield(NNlib, Symbol(args["activation"]));
    end

    return Chain(Conv((3, 3), 3 => 64, pad=SamePad()),
                 BatchNorm(64, act),
                 Conv((3, 3), 64 => 128, stride=(2, 2), pad=SamePad()),
                 BatchNorm(128, act),
                 Conv((3, 3), 128 => 128, stride=(2, 2), pad=SamePad()),
                 BatchNorm(128, act),
                 Conv((3, 3), 128 => 256, stride=(2, 2), pad=SamePad()),
                 BatchNorm(256, act),
                 x -> flatten(x),
                 Dropout(0.4),
                 Dense(4096, 10),
                 softmax);
end

function get_generator_v3(args)
    if args["activation"] in ["celu", "elu", "leakyrelu", "trelu"]
        # Now continue: We want to use Base.Fix2
        act = Fix2(getfield(NNlib, Symbol(args["activation"])), Float32(args["activation_alpha"]))
    else
        act = getfield(NNlib, Symbol(args["activation"]));
    end

    return Chain(x -> reshape(x, args["latent_dim"], :), 
                 Dense(args["latent_dim"], 4*4*256, act),
                 # Image is now (4)x(4)x(256)
                 x -> reshape(x, 4, 4, 256, :),
                 ConvTranspose((4, 4), 256=>128, act, stride=(2,2), pad=SamePad()),
                 # Image is now (8)x(8)x(128),
                 ConvTranspose((4, 4), 128=>128, act, stride=(2,2), pad=SamePad()),
                 # Image is now (16)x(16)x(128),
                 ConvTranspose((4, 4), 128 => 128, act, stride=(2,2), pad=SamePad()),
                 # Image is now (32)x(32)x(128),
                 Conv((3, 3), 128=>3, tanh, pad=SamePad()));
end
