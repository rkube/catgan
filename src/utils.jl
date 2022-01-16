#
#
export save_images

using FileIO
using Images


function save_images(x_fake, filename)
    # Flatten the individual samples xⁱ, generated by the Generator, side-by-side into a larger image.
    # x_fake: generator output, dims = (Width,Height,Channel,Batch)
    img_width, img_height, ch, num_images = size(x_fake)
    # Number of images per row
    num_rows = 10
    num_cols = num_images ÷ 10
    # img_arr will hold the output image
    img_array = zeros(RGB, img_height * num_rows, img_width * num_cols) 
    # Put the samples, addressed through the last dimension of X_cpu, column-wise into img_arr
    X_cpu = x_fake |> cpu;
    for row in 1:num_rows
       for col in 1:num_cols
           img_array[(row - 1) * img_width + 1:row * img_height, (col - 1) * img_width + 1: col * img_height] = colorview(RGB, permutedims(X_cpu[:, :, :, (row -1)* 10 + col], (3, 1, 2)))
       end
   end
   img_array = map(clamp01nan, img_array)

   save(filename, img_array)

end