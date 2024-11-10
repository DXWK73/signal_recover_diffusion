def get_1_dim_nconvd_size(input_size, output_size):
    def is_right(i, o, s, k, p):
        # if (o+2*p-k)%s == 0:
        return True if o == s*(i-1) - 2*p + k else False
        # else:
        #     return True if o == s*(i-1) - 2*p + k + (o+2*p-k)%s else False

    ans = []
    input_size = input_size[0]
    output_size = output_size[0]
    k_max = 10
    s_max = 10
    p_max = 1

    for k in range(1, k_max+1):
        for s in range(1, s_max+1):
            for p in range(0, p_max+1):
                if is_right(input_size, output_size, s, k, p):
                    ans.append(f"k={k}, s={s}, p={p}")
    return ans

def get_2_dim_nconvd_size():
    pass

if __name__ == '__main__':
    input = [250]
    output = [500]
    print(get_1_dim_nconvd_size(input, output))
