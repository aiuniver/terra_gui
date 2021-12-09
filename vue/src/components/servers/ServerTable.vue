<template>
  <table class="server-table">
    <thead>
      <tr>
        <th>Доменное имя</th>
        <th>IP адрес</th>
        <th>Имя пользователя</th>
        <th>SSH порт</th>
        <th>HTTP порт</th>
        <th>HTTPS порт</th>
        <th>Инструкция</th>
        <th>Состояние</th>
        <th style="min-width: 144px"></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>bl613490u.xyz</td>
        <td>255.255.255.255</td>
        <td>root</td>
        <td>22</td>
        <td>80</td>
        <td>422</td>
        <td class="clickable"><span>Открыть</span></td>
        <td>Не готов к работе</td>
        <td class="clickable">
          <i class="ci-icon ci-play_arrow"></i>
          <span>Запустить</span>
        </td>
      </tr>
      <tr>
        <td>bl613490u.xyz</td>
        <td>255.255.255.255</td>
        <td>root</td>
        <td>22</td>
        <td>80</td>
        <td>422</td>
        <td class="clickable"><span>Открыть</span></td>
        <td>Ожидание готовности</td>
        <td class="clickable"></td>
      </tr>
      <tr>
        <td>bl613490u.xyz</td>
        <td>255.255.255.255</td>
        <td>root</td>
        <td>22</td>
        <td>80</td>
        <td>422</td>
        <td class="clickable"><span>Открыть</span></td>
        <td>Готов к работе</td>
        <td class="clickable">
          <i class="ci-icon ci-redo"></i>
          <span>Обновить запуск</span>
        </td>
      </tr>
      <tr>
        <td>bl613490u.xyz</td>
        <td>255.255.255.255</td>
        <td>root</td>
        <td>22</td>
        <td>80</td>
        <td>422</td>
        <td class="clickable"><span>Открыть</span></td>
        <td class="error">
          Ошибка настройки
          <i class="ci-icon ci-info_circle_outline"></i>
        </td>
        <td class="clickable">
          <i class="ci-icon ci-play_arrow"></i>
          <span>Запустить</span>
        </td>
      </tr>
    </tbody>
  </table>
</template>

<script>
export default {
  name: 'ServerTable',
};
</script>

<style lang="scss" scoped>
.server-table {
  width: 100%;
  thead {
    color: #6c7883;
    font-size: 14px;
    height: 35px;
    th {
      font-weight: 400;
    }
  }
  tbody {
    font-size: 14px;
    tr {
      height: 55px;
      &:hover {
        background: #0e1621;
      }
    }
  }
  td,
  th {
    padding-left: 10px;
  }
  .clickable {
    color: #65b9f4;
    cursor: pointer;
    i {
      font-size: 16px;
      margin-right: 5px;
    }
    span {
      position: relative;
      vertical-align: text-bottom;
      &:hover::after {
        content: '';
        position: absolute;
        bottom: -5px;
        left: 0;
        width: 100%;
        height: 1px;
        background: #65b9f4;
      }
    }
  }
  .error {
    color: #ca5035;
    i {
      color: #65b9f4;
      font-size: 16px;
      vertical-align: middle;
    }
  }
}
</style>