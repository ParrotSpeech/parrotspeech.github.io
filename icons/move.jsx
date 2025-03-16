import React from "react"

export function MoveIcon(props) {
    return (
        <svg viewBox="0 0 75 24" fill="currentColor" {...props}>
            <path
                fillRule="evenodd"
                clipRule="evenodd"
                d="M2.08696 0C1.53346 0 1.00264 0.219875 0.611255 0.611255C0.219875 1.00264 0 1.53346 0 2.08696L0 21.913C0 22.4665 0.219875 22.9974 0.611255 23.3887C1.00264 23.7801 1.53346 24 2.08696 24H72C72.5535 24 73.0843 23.7801 73.4757 23.3887C73.8671 22.9974 74.087 22.4665 74.087 21.913V2.08696C74.087 1.53346 73.8671 1.00264 73.4757 0.611255C73.0843 0.219875 72.5535 0 72 0L2.08696 0ZM34.1259 7.5287C34.0864 7.57116 34.0387 7.60504 33.9855 7.6282C33.9324 7.65136 33.875 7.66331 33.817 7.6633H26.0348C27.3631 5.86122 29.4751 4.69565 31.8553 4.69565C34.2344 4.69565 36.3475 5.86122 37.6758 7.6633H36.2807C36.2066 7.66307 36.1334 7.64693 36.0661 7.61596C35.9987 7.585 35.9388 7.53994 35.8904 7.48383L35.3697 6.88383C35.3319 6.83974 35.285 6.80433 35.2322 6.78001C35.1795 6.7557 35.1221 6.74306 35.064 6.74296H35.0379C34.9827 6.7431 34.9281 6.75456 34.8775 6.77663C34.8269 6.79871 34.7813 6.83093 34.7437 6.8713L34.1259 7.5287ZM34.6873 9.64278C34.6139 9.64328 34.5412 9.62903 34.4735 9.60089C34.4057 9.57275 34.3443 9.53128 34.2929 9.47896L33.769 8.93322C33.7289 8.8923 33.681 8.85986 33.6282 8.83781C33.5753 8.81576 33.5185 8.80456 33.4612 8.80487C33.4037 8.8043 33.3467 8.81538 33.2937 8.83744C33.2406 8.8595 33.1925 8.89208 33.1523 8.93322L32.7037 9.40278C32.6278 9.47971 32.5373 9.54061 32.4374 9.58184C32.3376 9.62307 32.2305 9.64379 32.1224 9.64278H25.0497C24.8455 10.1801 24.7159 10.7429 24.6647 11.3155H31.343C31.4598 11.3155 31.5725 11.2717 31.6529 11.1923L32.2748 10.5934C32.3397 10.5309 32.4232 10.4913 32.5127 10.4807C32.5321 10.478 32.5516 10.4766 32.5711 10.4765H32.5972C32.6546 10.4753 32.7117 10.4861 32.7647 10.5082C32.8177 10.5303 32.8655 10.5632 32.905 10.6049L33.4299 11.1517C33.481 11.2041 33.5422 11.2456 33.6099 11.2738C33.6775 11.302 33.7501 11.3161 33.8233 11.3155H39.2734C39.222 10.7433 39.0924 10.1808 38.8883 9.64383L34.6873 9.64278ZM28.7635 15.1941C28.8206 15.1942 28.877 15.1825 28.9295 15.1599C28.9819 15.1373 29.0291 15.1042 29.0682 15.0626L29.6797 14.4136C29.717 14.3734 29.7622 14.3414 29.8125 14.3195C29.8628 14.2976 29.917 14.2863 29.9718 14.2863H29.9969C30.1117 14.2863 30.2223 14.3363 30.2995 14.425L30.8139 15.0177C30.912 15.1304 31.0529 15.1951 31.2 15.1951H38.4553C38.7313 14.6139 38.9295 13.9989 39.0449 13.3659H32.0817C32.0085 13.3656 31.9361 13.3496 31.8695 13.319C31.8029 13.2884 31.7436 13.2439 31.6957 13.1885L31.1802 12.5958C31.1426 12.5525 31.0961 12.5177 31.0439 12.4939C30.9917 12.4701 30.9349 12.4578 30.8776 12.4578C30.8202 12.4578 30.7635 12.4701 30.7113 12.4939C30.6591 12.5177 30.6126 12.5525 30.575 12.5958L30.1336 13.104C30.0627 13.186 29.9751 13.2518 29.8766 13.297C29.7782 13.3422 29.6711 13.3657 29.5628 13.3659H24.8932C25.009 13.9991 25.2073 14.6144 25.4828 15.1962L28.7635 15.1951V15.1941ZM29.087 17.0045C29.0128 17.0043 28.9395 16.9886 28.8717 16.9584C28.804 16.9283 28.7433 16.8842 28.6936 16.8292L28.1687 16.2449C28.1298 16.2019 28.0823 16.1675 28.0292 16.144C27.9762 16.1205 27.9189 16.1083 27.8609 16.1083C27.8029 16.1083 27.7455 16.1205 27.6925 16.144C27.6395 16.1675 27.592 16.2019 27.553 16.2449L27.1023 16.7468C27.0288 16.828 26.9391 16.8929 26.839 16.9373C26.7389 16.9817 26.6306 17.0046 26.521 17.0045H26.4908C27.1809 17.7331 28.0127 18.3129 28.935 18.7083C29.8573 19.1038 30.8507 19.3066 31.8543 19.3043C33.9725 19.3043 35.88 18.4195 37.2188 17.0045H29.087ZM9.45704 4.88557H6.26087V19.3043H8.7673V9.37774H8.90087L12.887 19.2616H14.759L18.744 9.39861H18.8776V19.3043H21.384V4.88557H18.1878L13.9064 15.3339H13.7374L9.45704 4.88557ZM44.8758 4.88557L48.6282 16.2344H48.7763L52.5224 4.88557H55.3941L50.3113 19.3043H47.087L42.0104 4.88557H44.8758ZM58.2845 4.88557V19.3043H67.7197V17.1151H60.8974V13.1791H67.177V10.9899H60.8974V7.07478H67.6633V4.88557H58.2845Z"
            />
        </svg>
    )
}
