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
            <th v-if="inputLayersNum" :colspan="inputLayersNum" ref="th_input">Исходные данные</th>
            <th v-if="outputTrueLayersNum" :colspan="outputTrueLayersNum" ref="th_true">Истинное значение</th>
            <th v-if="outputPredictLayersNum" :colspan="outputPredictLayersNum" ref="th_predict">Предсказание</th>
          </tr>

          <template v-for="({ initial_data, true_value, predict_value }, id) of predict">
            <template v-if="id === '1'">
              <tr :key="'tr1_' + id">
                <template v-for="(data, key, i) of initial_data">
                  <template v-if="!!i">
                    <th :key="'th1_' + key + i">{{ key }} {{i}}</th>
                  </template>
                  <template v-else>
                    <template v-for="(data, key, i) of true_value">
                      <th :key="'th2_' + key + i">{{ key }}</th>
                    </template>
                  </template>
                </template>

                <template v-for="(data, key, i) of true_value">
                  <th :key="'th2_' + key + i">{{ key }}</th>
                </template>

                <template v-for="(data, key, i) of predict_value">
                  <th :key="'th3_' + key + i">{{ key }}</th>
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
                    <th :style="{ height: `${keysHeight}px` }" :key="'th3' + i">{{ item.title }}</th>
                  </template>
                </template>
              </tr>
            </template>
          </template>
        </thead>

        <tbody class="t-table__body">
          <template v-for="({ initial_data, true_value, predict_value, tags_color }, id) of predict">
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
            </tr>
          </template>
        </tbody>
      </table>

      <table ref="original">
        <thead ref="orig_head" class="t-table__header">
          <tr>
            <th class="t-table__header--index" rowspan="3" ref="th_layer">Слой</th>
            <th v-if="inputLayersNum" :colspan="inputLayersNum" ref="th_input">Исходные данные</th>
            <th v-if="outputTrueLayersNum" :colspan="outputTrueLayersNum" ref="th_true">Истинное значение</th>
            <th v-if="outputPredictLayersNum" :colspan="outputPredictLayersNum" ref="th_predict">Предсказание</th>
            <th v-if="statLayersNum" :colspan="statLayersNum" ref="stats">Статистика примеров</th>
          </tr>

          <template v-for="({ initial_data, true_value, predict_value, statistic_values }, id) of predict">
            <template v-if="id === '1'">
              <tr :key="'tr1_' + id">
                <template v-for="(data, key, i) of initial_data">
                  <template v-if="!!i">
                    <th :key="'th1_' + key + i">{{ key }} {{i}}</th>
                  </template>
                  <template v-else>
                    <template v-for="(data, key, i) of true_value">
                      <th :key="'th2_' + key + i">{{ key }}</th>
                    </template>
                  </template>
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

              <tr ref="stat_headers" :key="'tr_' + id">
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
                    <Forms :data="item" :tags_color="tags_color" :layer="key" :type="type"/>
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
    fixation: {
      type: Boolean
    }
  },
  data: () => ({
    ops: {
      scrollPanel: {
        initialScrollY: 1
      },
      rail: {
        gutterOfSide: 0
      }
    },
    headHeight: 0,
    scrollLeft: 0,
    statsWidth: 0,
    keysHeight: 0,
    fixedWidth: 0
  }),
  computed: {
    colspan() {
      return Object.keys(this.predict[1].statistic_values[Object.keys(this.predict[1].statistic_values)[0]].data)
        .length
    },
    inputLayersNum() {
      return Object.keys(this.predict[1].initial_data).length
    },
    outputTrueLayersNum() {
      return Object.keys(this.predict[1].true_value).length
    },
    outputPredictLayersNum() {
      return Object.keys(this.predict[1].predict_value).length
    },
    statLayersNum() {
      return this.colspan * Object.keys(this.predict[1].statistic_values).length
    }
  },
  methods: {
    handleScroll(vert, horiz) {
      this.scrollLeft = horiz.scrollLeft
      this.scrollTop = vert.scrollTop
      this.tableResize()
    },
    tableResize() {
      this.statsWidth = this.$refs.stats.offsetWidth
      this.fixedWidth = this.$refs.original?.offsetWidth - this.statsWidth
      this.keysHeight = this.$refs.stat_headers[0].offsetHeight || 0
      this.headHeight = this.$refs.orig_head.offsetHeight + 1
    }
  },
  mounted() {
    this.$on('resize', this.tableResize)
    this.tableResize()
  },
  updated() {
    this.tableResize()
  }
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
    th {
      box-shadow: inset 1px 1px #000, 0 1px #000;
      padding: 0 5px;
      text-align: center;
      background-color: #242f3d;
      font-weight: normal;
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
    z-index: 1;
    thead {
      box-shadow: 1px 0 0 #000;
      z-index: 2;
    }
  }
}
</style>
