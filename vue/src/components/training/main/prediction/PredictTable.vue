<template>
  <div class="t-table">
    <scrollbar @handle-scroll="handleScroll" class="t-table__scroll" :ops="ops">
      <!-- <div class="t-table__layers" :style="{ left: `${scrollLeft}px`, paddingTop: `${headHeight}px` }">
        <div class="t-table__layers--head" :style="{ height: `${headHeight}px`, top: `${scrollTop}px` }">Слой</div>
        <div class="t-table__layers--item"
        :style="{ height: `${cellHeight}px` }"
        v-for="(val, id) of predict"
        :key="'thl_'+id">
          {{ id }}
        </div>
      </div> -->
      <table v-show="fixation" class="t-table__fixed" :style="{ left: `${scrollLeft}px`, width: `${fixedWidth}px` }">
        <thead class="t-table__header" :style="{ height: `${headHeight}px` }">
          <tr>
            <th class="t-table__header--index" rowspan="3" ref="th_layer">Слой</th>
            <th v-if="getDataColSpan('initial_data')" :colspan="getDataColSpan('initial_data')" ref="th_input">
              Исходные данные
            </th>
            <th v-if="getDataColSpan('true_value')" :colspan="getDataColSpan('true_value')" ref="th_true">
              Истинное значение
            </th>
            <th v-if="getDataColSpan('predict_value')" :colspan="getDataColSpan('predict_value')" ref="th_predict">
              Предсказание
            </th>
          </tr>

          <template v-for="({ initial_data, true_value, predict_value }, id) of predict">
            <template v-if="id === '1'">
              <tr :key="'f1tr1_' + id">
                <template v-for="(data, key, i) of initial_data">
                  <template>
                    <th :colspan="getLayerColSpan(initial_data, key)" :key="'f1th1_' + JSON.stringify(key) + i">
                      {{ key }}
                    </th>
                  </template>
                </template>

                <template v-for="(data, key, i) of true_value">
                  <th :colspan="getLayerColSpan(true_value, key)" :key="'f1th2_' + JSON.stringify(key) + i">
                    {{ key }}
                  </th>
                </template>

                <template v-for="(data, key, i) of predict_value">
                  <th :colspan="getLayerColSpan(predict_value, key)" :key="'f1th3_' + JSON.stringify(key) + i">
                    {{ key }}
                  </th>
                </template>
              </tr>

              <tr :key="'f2tr_' + id">
                <template v-for="key of initial_data">
                  <template v-for="(item, i) in key.data">
                    <th :key="'f2th1' + i + JSON.stringify(key)">{{ item.title }}</th>
                  </template>
                </template>

                <template v-for="key of true_value">
                  <template v-for="(item, i) in key.data">
                    <th :key="'f2th2' + i">{{ item.title }}</th>
                  </template>
                </template>

                <template v-for="key of predict_value">
                  <template v-for="(item, i) in key.data">
                    <th :style="{ height: `${keysHeight}px` }" :key="'f2th3' + i">{{ item.title }}</th>
                  </template>
                </template>
              </tr>
            </template>
          </template>
        </thead>

        <tbody class="t-table__body">
          <template v-for="({ initial_data, true_value, predict_value, tags_color }, id) of predict">
            <tr :key="'frows_' + id">
              <td>
                {{ id }}
              </td>

              <template v-for="({ type, data, update }, key) in initial_data">
                <template v-for="(item, i) of data">
                  <td :key="'finitial_layer_' + i + JSON.stringify(key)">
                    <Forms :data="item" :tags_color="tags_color" :layer="key" :update="update" :type="type" />
                  </td>
                </template>
              </template>

              <template v-for="({ type, data }, key) in true_value">
                <template v-for="(item, i) of data">
                  <td :key="'ftrue_layer_' + i + JSON.stringify(key)">
                    <Forms :data="item" :tags_color="tags_color" :layer="key" :type="type" />
                  </td>
                </template>
              </template>

              <template v-for="({ type, data }, key) in predict_value">
                <template v-for="(item, i) of data">
                  <td :key="'fpredict_layer_' + i + JSON.stringify(key)">
                    <Forms :data="item" :tags_color="tags_color" :layer="key" :type="type" />
                  </td>
                </template>
              </template>
            </tr>
          </template>
        </tbody>
      </table>

      <table ref="original">
        <thead ref="orig_head" class="t-table__header">
          <tr>
            <th class="t-table__header--index" rowspan="3" ref="th_layer">Слой</th>
            <th v-if="getDataColSpan('initial_data')" :colspan="getDataColSpan('initial_data')" ref="th_input">
              Исходные данные
            </th>
            <th v-if="getDataColSpan('true_value')" :colspan="getDataColSpan('true_value')" ref="th_true">
              Истинное значение
            </th>
            <th v-if="getDataColSpan('predict_value')" :colspan="getDataColSpan('predict_value')" ref="th_predict">
              Предсказание
            </th>
            <th v-if="getDataColSpan('statistic_values')" :colspan="getDataColSpan('statistic_values')" ref="stats">
              Статистика примеров
            </th>
          </tr>

          <template v-for="({ initial_data, true_value, predict_value, statistic_values }, id) of predict">
            <template v-if="id === '1'">
              <tr :key="'1tr1_' + id">
                <template v-for="(data, key, i) of initial_data">
                  <template>
                    <th :colspan="getLayerColSpan(initial_data, key)" :key="'1th1_' + JSON.stringify(key) + i">
                      {{ key }}
                    </th>
                  </template>
                </template>

                <template v-for="(data, key, i) of true_value">
                  <th :colspan="getLayerColSpan(true_value, key)" :key="'1th2_' + JSON.stringify(key) + i">
                    {{ key }}
                  </th>
                </template>

                <template v-for="(data, key, i) of predict_value">
                  <th :colspan="getLayerColSpan(predict_value, key)" :key="'1th3_' + JSON.stringify(key) + i">
                    {{ key }}
                  </th>
                </template>

                <template v-for="(data, key, i) of statistic_values">
                  <th :colspan="getLayerColSpan(statistic_values, key)" :key="'1th4_' + JSON.stringify(key) + i">
                    {{ key }}
                  </th>
                </template>
              </tr>

              <tr ref="stat_headers" :key="'2tr_' + id">
                <template v-for="(data, key) of initial_data">
                  <template v-for="(item, i) in data.data">
                    <th :key="'2th1' + i + +key">{{ item.title }}</th>
                  </template>
                </template>

                <template v-for="(data, key) of true_value">
                  <template v-for="(item, i) in data.data">
                    <th :key="'2th2' + i + key">{{ item.title }}</th>
                  </template>
                </template>

                <template v-for="(data, key) of predict_value">
                  <template v-for="(item, i) in data.data">
                    <th :key="'2th3' + i + key">{{ item.title }}</th>
                  </template>
                </template>

                <template v-for="(data, key) of statistic_values">
                  <template v-for="(item, i) in data.data">
                    <th class="t-table__header--static" :key="'2th4' + i + key">{{ item.title }}</th>
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

              <template v-for="({ type, data }, key) in initial_data">
                <template v-for="(item, i) of data">
                  <td :key="'initial_layer_' + i + JSON.stringify(key) + update">
                    <Forms :data="item" :tags_color="tags_color" :layer="key" :update="update" :type="type" />
                  </td>
                </template>
              </template>

              <template v-for="({ type, data }, key) in true_value">
                <template v-for="(item, i) of data">
                  <td :key="'true_layer_' + i + JSON.stringify(key) + update">
                    <Forms :data="item" :tags_color="tags_color" :layer="key" :update="update" :type="type" />
                  </td>
                </template>
              </template>

              <template v-for="({ type, data }, key) in predict_value">
                <template v-for="(item, i) of data">
                  <td :key="'predict_layer_' + i + JSON.stringify(key) + update">
                    <Forms :data="item" :tags_color="tags_color" :layer="key" :update="update" :type="type" />
                  </td>
                </template>
              </template>

              <template v-for="({ type, data }, key) in statistic_values">
                <template v-for="(item, i) of data">
                  <td :key="'statistic_layer_' + i + JSON.stringify(key) + update">
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
    fixation: {
      type: Boolean,
    },
    update: String,
  },
  data: () => ({
    ops: {
      scrollPanel: {
        initialScrollY: 1,
      },
    },
    headHeight: 0,
    scrollLeft: 0,
    statsWidth: 0,
    keysHeight: 0,
    fixedWidth: 0,
  }),
  methods: {
    handleScroll(vert, horiz) {
      this.scrollLeft = horiz.scrollLeft;
      this.scrollTop = vert.scrollTop;
      this.tableResize();
    },
    tableResize() {
      this.statsWidth = this.$refs?.stats?.offsetWidth || 0;
      this.fixedWidth = (this.$refs?.original?.offsetWidth || 0) - this.statsWidth;
      this.keysHeight = this.$refs?.stat_headers?.[0]?.offsetHeight || 0;
      this.headHeight = this.$refs?.orig_head?.offsetHeight || 0;
    },
    getLayerColSpan(data, key) {
      return data[key].data.length;
    },
    getDataColSpan(data) {
      let layers = 0;
      Object.keys(this.predict[1][data]).forEach(item => {
        layers += this.predict[1][data][item].data.length;
      });
      return layers;
    },
  },
  mounted() {
    this.$on('resize', this.tableResize);
    this.tableResize();
  },
  updated() {
    this.tableResize();
  },
};
</script>

<style lang="scss" scoped>
.t-table {
  position: relative;
  height: 600px;
  table {
    text-align: center;
    border-radius: 4px;
  }
  &__header {
    position: sticky;
    top: 0;
    box-shadow: 1px 0 0 #000;
    z-index: 4;
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
      padding: 10px;
      text-align: center;
    }
  }
  &__fixed {
    background-color: #17212b;
    position: absolute;
    left: 0;
    z-index: 5;
    thead {
      box-shadow: 1px 0 0 #000;
    }
  }
}
</style>
