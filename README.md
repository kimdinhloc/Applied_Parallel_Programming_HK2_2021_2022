## LẬP TRÌNH SONG SONG ỨNG DỤNG - HK2/21-22 - FIT- HCMUS
### BACKGROUND REMOVAL WITH GRABCUT AND SUPERPIXELS

### GIẢNG VIÊN HƯỚNG DẪN
Thầy: Trần Trung Kiên - Nguyễn Thái Vũ

### THÀNH VIÊN

| MSSV | Họ tên | Tài khoản Github |
| --- | --- | --- |
| 1712568 | KIM ĐÌNH LỘC | [kimdinhloc](https://github.com/kimdinhloc) |
| 1712787 | NGUYỄN VĂN THÌN | [thinnguyenqb](https://github.com/thinnguyenqb) |
| 1712898 | TRẦN VIỆT VĂN | [tranvietvan](https://github.com/tranvietvan) |

### MỤC TIÊU ĐỒ ÁN
Đồ án hướng đến việc ứng dụng của segmentation trong xử lý ảnh để tách nền các object dựa trên các Super Pixel.
### THÔNG TIN ĐỒ ÁN
- Báo cáo: <a href="https://colab.research.google.com/github/kimdinhloc/Applied_Parallel_Programming_HK2_2021_2022/blob/main/Report.ipynb">Colab</a>
- Tài liệu tham khảo: 
  - <a href="https://drive.google.com/drive/folders/1gjfWWeSdgjei1dVaIRH1wYsrM-IdtyVQ?usp=sharing">Google Drive</a>
### KẾ HOẠCH THỰC HIỆN

<table>
<thead>
  <tr>
   <th>Tuần</th>
   <th>Bắt Đầu</th>
   <th>Kết Thúc</th>
   <th>Công Việc</th>
   <th>Thành Viên Thực Hiện</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>1</td>
    <td>24/03/2022</td>
    <td>30/04/2022</td>
    <td>
      - Mỗi người tìm hiểu và đưa ra một đề xuất đề tài .
      <br>
      - Thống nhất đề tài đồ án là Remove Background.
     <br>
     - Tìm hiểu thuật toán để Remove Background.
   </td>
   <td>Cả nhóm</td>
  </tr>
  <tr>
    <td rowspan="3">2</td>
    <td rowspan="3">31/03/2022</td>
    <td rowspan="3">06/04/2022</td>
    <td> 
    - Mỗi thành viên tự cài đặt thuật toán tuần tự để lựa chọn mô hình thích hợp.
    </td>
    <td>Cả nhóm</td>
  </tr>

  <tr>
    <td> 
    - Soạn báo cáo, viết notebook.
    </td>
    <td>Thìn Nguyễn</td>
  </tr>
  <tr>
    <td> 
    - Báo cáo lần 1.
    </td>
    <td>Đình Lộc</td>
  </tr>

   <tr>
    <td>3</td>
    <td>07/04/2022</td>
    <td>14/04/2022</td>
    <td>- Xây dựng mô hình và các bước để Remove background</td>
    <td>Cả nhóm</td>
   </tr>
   <tr>
   <td rowspan="3">4</td>
    <td rowspan="3">15/04/2022</td>
    <td rowspan="3">20/04/2022</td>
    <td>
    - Cài đặt quá trình gom cụm của SLIC Superpixels
   </td>
    <td>Cả nhóm</td>
  </tr>

  <tr>
    <td>
    - Tìm hiểu Objects Detection.
    </br>
    - Cài đặt thuật toán xác định biên cạnh.
   </td>
    <td>Việt Văn</td>
  </tr>

  <tr>
    <td>
    - Bổ sung, chỉnh sửa file README, soạn báo cáo.
    </br>
    
   </td>
    <td>Thìn Nguyễn</td>
  </tr>

   
   <tr>
   <td rowspan="2">5,6</td>
    <td rowspan="2">21/04/2022</td>
    <td rowspan="2">12/05/2022</td>
    <td>
    - Tìm hiểu viết kernel bằng Numba
    </br>
    - Viết code ở host
   </td>
    <td>Cả nhóm</td>
  </tr>

  <tr>
    <td>
    - Chỉnh sửa mô hình.
   </td>
    <td>Đình Lộc</td>
  </tr>
  
   <tr>
   <td rowspan="1">7</td>
    <td rowspan="1">13/05/2022</td>
    <td rowspan="1">18/05/2022</td>
    <td>
     - Fix bug ở host
    </br>
    - Tìm hiểu chi tiết về Memory management trong numba
   </td>
    <td>Cả nhóm</td>
  </tr>

   <tr>
   <td rowspan="1">8</td>
    <td rowspan="1">18/05/2022</td>
    <td rowspan="1">21/05/2022</td>
    <td>
    - Viết kernel
    </br>
    - Song song hoá lần 1
   </td>
    <td>Cả nhóm</td>
  </tr>
   <tr>
    <td rowspan="1">9,10</td>
        <td rowspan="1">23/05/2022</td>
        <td rowspan="1">06/06/2022</td>
        <td>
        - Hoàn thiện các hàm jit trong SLIC
        </br>
        - Song song hoá các hàm trong SLIC
        </br>
        - Chuẩn bị, viết report cho buổi vấn đáp
       </td>
        <td>Cả nhóm</td>
 </tr>
    </tbody>
</table>
