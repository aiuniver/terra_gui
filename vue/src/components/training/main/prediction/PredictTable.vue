<template>
  <div class="t-table">
    <scrollbar>
      <table>
        <thead class="t-table__header">
          <tr>
            <th class="t-table__header--index" rowspan="3">Слой</th>
            <th>Исходные данные</th>
            <th>Истинное значение</th>
            <th>Предсказание</th>
            <th :colspan="colspan">Статистика примеров</th>
          </tr>

          <template v-for="({ initial_data, true_value, predict_value, statistic_values }, id) of predict">
            <template v-if="id === '1'">
              <tr :key="'tr1_' + id">
                <template v-for="(data, key, i) of initial_data">
                  <th :key="'th1_' + key + i">{{ key }}</th>
                </template>

                <template v-for="(data, key, i) of true_value">
                  <th :key="'th2_' + key + i">{{ key }}</th>
                </template>

                <template v-for="(data, key, i) of predict_value">
                  <th :key="'th3_' + key + i">{{ key }}</th>
                </template>

                <template v-for="(data, key, i) of statistic_values">
                  <th :colspan="colspan" :key="'th4_' + key + i">{{ key }}</th>
                </template>
              </tr>

              <tr :key="'tr_' + id">
                <template v-for="key of initial_data">
                  <template v-for="(item, i) in key.data">
                    <th :key="'th1' + i + key">{{ item.title }}</th>
                  </template>
                </template>

                <template v-for="key of true_value">
                  <template v-for="(item, i) in key.data">
                    <th :key="'th2' + i">{{ item.title }}</th>
                  </template>
                </template>

                <template v-for="key of predict_value">
                  <template v-for="(item, i) in key.data">
                    <th :key="'th3' + i">{{ item.title }}</th>
                  </template>
                </template>

                <template v-for="key of statistic_values">
                  <template v-for="(item, i) in key.data">
                    <th class="t-table__header--static" :key="'th4' + i">{{ item.title }}</th>
                  </template>
                </template>
              </tr>
            </template>
          </template>
        </thead>

        <tbody class="t-table__body">
          <template v-for="({ initial_data, true_value, predict_value, statistic_values, tags_color }, id) of predict">
            <tr :key="'rows_' + id">
              <td>
                {{ id }}
              </td>

              <template v-for="({ type, data, update }, key) in initial_data">
                <template v-for="(item, i) of data">
                  <td :key="'initial_layer_' + i">
                    <Forms :data="item" :tags_color="tags_color" :layer="key" :update="update" :type="type" />
                  </td>
                </template>
              </template>

              <template v-for="({ type, data }, key) in true_value">
                <template v-for="(item, i) of data">
                  <td :key="'true_layer_' + i">
                    <Forms :data="item" :tags_color="tags_color" :layer="key" :type="type" />
                  </td>
                </template>
              </template>

              <template v-for="({ type, data }, key) in predict_value">
                <template v-for="(item, i) of data">
                  <td :key="'predict_layer_' + i">
                    <Forms :data="item" :tags_color="tags_color" :layer="key" :type="type" />
                  </td>
                </template>
              </template>

              <template v-for="{ type, data } in statistic_values">
                <template v-for="(item, i) of data">
                  <td :key="'statistic_layer_' + i">
                    <Forms :data="item" :type="type" />
                  </td>
                </template>
              </template>
            </tr>
          </template>
        </tbody>
      </table>
    </scrollbar>
  </div>
</template>

<script>
import Forms from './components/Forms';
export default {
  name: 'TextTableTest',
  components: {
    Forms,
  },
  props: {
    show: Boolean,
    predict: {
      type: Object,
      default: () => ({}),
    },
  },
  data: () => ({
    ops: {
      bar: { background: '#17212b' },
      scrollPanel: {
        scrollingX: false,
        scrollingY: true,
      },
    },
  }),
  computed: {
    colspan() {
      return Object.keys(this.predict[1].statistic_values[Object.keys(this.predict[1].statistic_values)[0]].data)
        .length;
    },
  },
};
</script>

<style lang="scss" scoped>
.t-table {
  position: relative;
  height: 600px;
  table {
    width: 100%;
    text-align: center;
    border-radius: 4px;
  }
  &__header {
    position: sticky;
    top: 0;
    z-index: 1;
    th {
      box-shadow: inset 1px 1px #000, 0 1px #000;
      padding: 0 5px;
      text-align: center;
      background-color: #242f3d;
      font-weight: normal;
      min-width: 220px;
    }
    &--index {
      min-width: 70px !important;
    }
    &--static {
      min-width: 70px !important;
    }
  }
  &__body {
    td {
      border: 1px solid #0e1621;
      padding: 10px;
      text-align: center;
    }
  }
}
</style>
