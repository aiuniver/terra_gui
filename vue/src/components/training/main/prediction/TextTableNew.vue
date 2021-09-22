<template>
  <div class="t-table">
    <scrollbar>
      <slot name="header"></slot>
      <div class="t-table__body">
        <div
          class="t-table__rows"
          v-for="({ initial_data, true_value, predict_value, statistic_values }, id) of predict"
          :key="'rows_' + id"
        >
          <div v-if="id === '1'" class="t-table__title t-table__title--index">Слой</div>
          <div class="t-table__index">{{ id }}</div>
          <div v-if="isEmpty(initial_data)" class="t-table__col">
            <div v-if="id === '1'" class="t-table__title t-table__title--one">Исходные данные</div>
            <div class="t-table__row">
              <div class="t-table__col" v-for="({ type, data }, key, i) in initial_data" :key="`initial ${i}`">
                <div v-if="id === '1'" class="t-table__title t-table__title--two">{{ key }}</div>
                <div class="t-table__row">
                  <div class="t-table__col" v-for="(item, i) of data" :key="`initial layer ${i}`">
                    <div v-if="id === '1'" class="t-table__title t-table__title--three">{{ item.title }}</div>
                    <div class="t-table__row t-table__row--center">
                      <Forms :data="item" :type="type" :key="`initial data ${i}`" />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div v-if="isEmpty(true_value)" class="t-table__col">
            <div v-if="id === '1'" class="t-table__title t-table__title--one">Истинное значение</div>
            <div class="t-table__row">
              <div class="t-table__col" v-for="({ type, data }, key, i) in true_value" :key="`true ${i}`">
                <div v-if="id === '1'" class="t-table__title t-table__title--two">{{ key }}</div>
                <div class="t-table__row">
                  <div class="t-table__col" v-for="(item, i) of data" :key="`true layer ${i}`">
                    <div v-if="id === '1'" class="t-table__title t-table__title--three">{{ item.title }}</div>
                    <div class="t-table__row t-table__row--center">
                      <Forms :data="item" :type="type" :key="`true data ${i}`" />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div v-if="isEmpty(predict_value)" class="t-table__col">
            <div v-if="id === '1'" class="t-table__title t-table__title--one">Предсказание</div>
            <div class="t-table__row">
              <div class="t-table__col" v-for="({ type, data }, key, i) in predict_value" :key="`predict ${i}`">
                <div v-if="id === '1'" class="t-table__title t-table__title--two">{{ key }}</div>
                <div class="t-table__row">
                  <div class="t-table__col" v-for="(item, i) of data" :key="`predict layer ${i}`">
                    <div v-if="id === '1'" class="t-table__title t-table__title--three">{{ item.title }}</div>
                    <div class="t-table__row t-table__row--center">
                      <Forms :data="item" :type="type" :key="`predict data ${i}`" />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div v-if="isEmpty(statistic_values)" class="t-table__col">
            <div v-if="id === '1'" class="t-table__title t-table__title--one">Статистика примеров</div>
            <div class="t-table__row">
              <div class="t-table__col" v-for="({ type, data }, key, i) in statistic_values" :key="`statistic ${i}`">
                <div v-if="id === '1'" class="t-table__title t-table__title--two">{{ key }}</div>
                <div class="t-table__row">
                  <div class="t-table__col" v-for="(item, i) of data" :key="`statistic layer ${i}`">
                    <div v-if="id === '1'" class="t-table__title t-table__title--three">{{ item.title }}</div>
                    <div class="t-table__row t-table__row--center t-table__row--table">
                      <Forms :data="item" :type="type" :key="`statistic data ${i}`" />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <!-- <div v-if="isEmpty(tags_color)" class="t-table__col">
            <div v-if="id === '1'" class="t-table__title t-table__title--one">Статистика примеров</div>
            <div class="t-table__row">
              <div class="t-table__col" v-for="({ type, data }, key, i) in tags_color" :key="`tags ${i}`">
                <div v-if="id === '1'" class="t-table__title t-table__title--two">{{ key }}</div>
                <div class="t-table__row">
                  <div class="t-table__col" v-for="(item, i) of data" :key="`tags layer ${i}`">
                    <div v-if="id === '1'" class="t-table__title t-table__title--three">{{ item.title }}</div>
                    <div class="t-table__row t-table__row--center t-table__row--table">
                      {{ type}}
                      <Forms :data="item" :type="type" :key="`tags data ${i}`" />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div> -->
        </div>
      </div>
    </scrollbar>
  </div>
</template>

<script>
import Forms from './components/Forms';
// import TableImage from './components/TableImage';
// import TableText from './components/TableText';
// import TableStatisticText from './components/TableStatisticText';
// import TableTag from './components/TableTag.vue';
// import TableAudio from './components/TableAudio';
// import Embed from 'v-video-embed/src/embed';
export default {
  name: 'TextTableTest',
  components: {
    Forms,
    // TableStatisticText,
  },
  props: {
    show: Boolean,
    predict: {
      type: Object,
      default: () => ({}),
    },
  },
  data: () => ({}),
  computed: {},
  methods: {
    isEmpty(obj) {
      return Object.keys(obj).length;
    },
  },
  created() {
    console.log('Predict', this.predict);
  },
};
</script>

<style lang="scss" scoped>
.t-table {
  position: relative;
  height: 600px;
  border: 1px solid #0e1621;
  &__body {
    padding: 66px 0 0 0;
  }
  &__rows {
    position: relative;
    display: flex;
    width: 100%;
    border-bottom: 1px solid #0e1621;
  }
  &__index {
    width: 70px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-right: 1px solid #0e1621;
  }
  &__title {
    display: flex;
    justify-content: center;
    position: absolute;
    align-items: center;
    background-color: #242f3d;
    border: 1px solid #0e1621;
    padding: 0 10px;
    width: 100%;
    z-index: 5;
    &--one {
      top: -66px;
    }
    &--two {
      top: -44px;
      font-size: 14px;
    }
    &--three {
      top: -22px;
      font-size: 14px;
    }
    &--index {
      width: 70px;
      height: 67px;
      top: -66px;
    }
  }
  &__row {
    display: flex;
    min-width: 200px;

    &--table {
      min-width: 100px;
    }
    flex: 1;
    &--center {
      align-items: center;
      justify-content: center;
      border-right: 1px solid #0e1621;
      border-left: 1px solid #0e1621;
    }
  }
  &__col {
    display: flex;
    flex-direction: column;
    // flex: 1 0 auto;
    position: relative;
    &--main {
    }
  }
}
</style>
