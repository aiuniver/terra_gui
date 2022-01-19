<template>
  <div class="t-statistic-table">
    <table>
    <thead class="t-statistic-table__header">
      
      <th v-for="(headline, key, i) in value[0]" :key="JSON.stringify(headline + key + i)">
        {{ key }}
      </th>

    </thead>
    <tbody class="t-statistic-table__body">
      <template v-for="(data, idx) in value">
        <tr :key="JSON.stringify('1tr' + data + idx)">
          <td>
            {{ data[Object.keys(data)[0]] }}
          </td>
          <td>
            {{ data[Object.keys(data)[1]] }}
          </td>
          <td>
            {{ data[Object.keys(data)[2]] }}
          </td>
          <td>
            <Forms :data="data[Object.keys(data)[3]]" type="str" :update="JSON.stringify('update' + data + idx)"/>
          </td>
        </tr>
      </template>
     
    </tbody>
  </table>
  </div>
</template>

<script>
import Forms from './Forms';

export default {
  name: 't-statistic-table',
  components:{
    Forms
  },
  props: {
    title: String,
    value: [Object, Array]
  }
};
</script>

<style lang="scss" scoped>
 table {
  text-align: center;
  border-radius: 4px;
}
.t-statistic-table {
  position: relative;
  &__header {
    th {
      box-shadow: inset 1px 1px #000, 0 1px #000;
      padding: 0 5px;
      text-align: center;
      background-color: #242f3d;
      font-weight: normal;
      white-space: nowrap;
    }
  }
  &__body {
    td {
      border: 1px solid #0e1621;
      padding: 5px;
      text-align: center;
    }
  }
}
</style>
