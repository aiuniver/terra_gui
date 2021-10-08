<template>
  <div class="t-table">
    <p class="t-table__title">{{ graph_name }}</p>
    <table class="t-table__table">
      <thead>
        <tr class="bold">
          <th rowspan="2">Класс</th>
          <th colspan="4">Статистические данные</th>
        </tr>
        <tr>
          <th>Точность</th>
          <th>Чувствительность</th>
          <th>F1-мера</th>
          <th>Количество</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(value, key, idx) in plot_data" :key="idx">
          <td>{{ value['Класс'] }}</td>
          <td>{{ value['Точность'] }}</td>
          <td>{{ value['Чувствительность'] }}</td>
          <td>{{ value['F1-мера'] }}</td>
          <td>{{ value['Количество'] }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script>
export default {
  name: 't-table',
  props: {
    id: Number,
    plot_data: Array,
    graph_name: String,
    type: String
  },
  data: () => ({

  }),
};
</script>

<style lang="scss" scoped>
.t-table {
  &__title {
    font-size: 14px;
    line-height: 17px;
    font-weight: 600;
    text-align: center;
    color: #a7bed3;
    margin: 10px auto;
  }
  .bold th {
    font-weight: 600;
    line-height: 16px;
  }
  &__table {
    font-size: 12px;
    line-height: 14px;
    border: 2px solid #242f3d;
    border-radius: 4px;
    overflow: hidden;
    thead {
      color: #a7bed3;
      background: #242f3d;
      height: 45px;
      th {
        text-align: center;
        font-weight: 400;
        min-width: 88px;
        padding: 2px 3px;
        border: 1px solid black;
      }
    }
    tbody {
      line-height: 16px;
      background: #17212b;
      td {
        text-align: center;
        padding: 4px 0;
        border: 1px solid black;
      }
      tr:nth-last-child(-n + 4) {
        background: #242f3d;
      }
    }
  }
}
</style>
