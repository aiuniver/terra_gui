<template>
  <table>
    <thead>
      <tr class="outputs_heads">
        <th rowspan="2">Эпоха</th>
        <th rowspan="2">Время<br />(сек.)</th>
        <th colspan="4">output_1</th>
      </tr>
      <tr class="callbacks_heads">
        <th>accuracy</th>
        <th>val_accuracy</th>
        <th>loss</th>
        <th>val_loss</th>
      </tr>
    </thead>
    <tbody>
      <tr v-for="({data:{output_1:{accuracy, loss, val_accuracy, val_loss}}, time, number}, i) of epochs" :key="'epoch_' + i">
        <td class="epoch_num">{{ number }}</td>
        <td>{{ time }}</td>
        <td class="value"><span>{{ accuracy | int }}</span><i>.</i>{{ accuracy | drob }}</td>
        <td class="value"><span>{{ loss | int }}</span><i>.</i>{{ loss | drob }}</td>
        <td class="value"><span>{{ val_accuracy | int }}</span><i>.</i>{{ val_accuracy | drob }}</td>
        <td class="value"><span>{{ val_loss | int }}</span><i>.</i>{{ val_loss | drob }}</td>

      </tr>
    </tbody>
    <tfoot>
      <tr>
        <th colspan="6">{{ summary }}</th>
      </tr>
    </tfoot>
  </table>
</template>

<script>
export default {
  name: "TTable",
  props: {
    summary: {
      type: String,
      default: ''
    },
    epochs: {
      type: Array,
      default: () => [],
    },
  },
  filters: {
    int(val) {
      return ~~val
    },
    drob(val) {
      return (val%1).toFixed(3).slice(2)
    }
  }
};
</script>

<style lang="scss" scoped>
table {
  user-select: none;
  border-radius: 4px;
  box-shadow: 0 2px 10px 0 rgb(0 0 0 / 25%);
  border-collapse: collapse;
  border-spacing: 0;
  th {
    background: #242f3d;
    line-height: 1.25;
    padding: 10px 15px;
    color: #a7bed3;
    font-size: 0.875rem;
    text-align: center;
    border: 1px solid #0e1621;
  }
  td {
    line-height: 1.25;
    padding: 10px 15px;
    color: #a7bed3;
    font-size: 0.875rem;
    text-align: center;
    border: 1px solid #0e1621;
    &.value {
      color: #fff;
      font-size: 0.75rem;
      text-align: right;
      font-family: monospace, sans-serif;
      i {
        color: #4b6780;
        font-style: normal;
      }
      span {
        color: #65b9f4;
      }
    }
  }
}
</style>