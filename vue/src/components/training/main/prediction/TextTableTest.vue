<template>
  <div class="table" v-if="show">
    <div class="table__columns">
      <div class="table__column">
        <div class="table__item title size-1 layer-title">Слой</div>
        <div
          class="table__item"
          v-for="index in Object.keys(predict.initial_value[Object.keys(predict.initial_value)[0]]).length"
          :key="index"
        >
          {{ index }}
        </div>
      </div>
      <div class="table__column">
        <div class="table__column" v-if="predict.initial_value">
          <div class="table__item title size-2">Исходные данные</div>
          <div class="table__row">
            <div class="table__column" v-for="(value, input_name) in predict.initial_value" :key="input_name">
              <div class="table__item title size-2">{{ input_name }}</div>
              <div class="table__item" v-for="(input_val, key) in value" :key="key">
                <TableImage v-if="input_val.type == 'image'" :image="input_val" />
                <TableText v-if="input_val.type === 'str' || input_val.type === 'number'" :data="input_val" />
                <Embed v-if="input_val.type === 'video'" :src="input_val.data"></Embed>
                <TableAudio v-if="input_val.type === 'audio'"/>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="table__column">
        <div class="table__column" v-if="predict.true_value">
          <div class="table__item title size-2">Истинное значение</div>
          <div class="table__row">
            <div class="table__column" v-for="(value, input_name) in predict.true_value" :key="input_name">
              <div class="table__item title size-2">{{ input_name }}</div>
              <div class="table__item" v-for="(input_val, key) in value" :key="key">
                <TableImage v-if="input_val.type == 'image'" :image="input_val" />
                <TableText v-if="input_val.type === 'str' || input_val.type === 'number'" :data="input_val" />
                <Embed v-if="input_val.type === 'video'" :src="input_val.data"></Embed>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="table__column">
        <div class="table__column" v-if="predict.predict_value">
          <div class="table__item title size-2">Предсказание</div>
          <div class="table__row">
            <div class="table__column" v-for="(value, input_name) in predict.predict_value" :key="input_name">
              <div class="table__item title size-2">{{ input_name }}</div>
              <div class="table__item" v-for="(input_val, key) in value" :key="key">
                <TableImage v-if="input_val.type == 'image'" :image="input_val" />
                <TableText v-if="input_val.type === 'str' || input_val.type === 'number'" :data="input_val" />
                <Embed v-if="input_val.type === 'video'" :src="input_val.data"></Embed>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="table__column">
        <div class="table__column" v-if="predict.statistic_values">
          <div class="table__item title size-2">Статистика примеров</div>
          <div class="table__row">
            <div class="table__column" v-for="(value, input_name) in predict.statistic_values" :key="input_name">
              <div class="table__item title size-2">{{ input_name }}</div>
              <div class="table__item" v-for="(input_val, key) in value" :key="key">
                <TableImage v-if="input_val.type == 'image'" :image="input_val" />
                <TableText v-if="input_val.type === 'str' || input_val.type === 'number'" :data="input_val" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import TableImage from '@/components/training/main/prediction/components/TableImage.vue';
import TableText from '@/components/training/main/prediction/components/TableText.vue';
import TableAudio from "../audio/TableAudio";
import Embed from 'v-video-embed/src/embed';
export default {
  name: 'TextTableTest',
  components: {
    TableImage,
    TableText,
    TableAudio,
    Embed,
  },
  props: {
    show: Boolean,
    predict: {
      type: Object,
      default: () => ({}),
    },
  },
  data: () => ({}),
  created() {
    console.log('Predict', this.predict);
  },
};
</script>

<style lang="scss" scoped>
$bgGray: #242f3d;
$border: #0e1621;

.table {
  width: 100%;
  border: 1px solid $bgGray;
  border-radius: 3px;
  &__columns {
    display: flex;
    flex-direction: row;
  }
  &__row {
    display: flex;
    height: 100%;
  }
  &__column {
    display: flex;
    flex-direction: column;
    width: fill-available;
    min-height: 100%;
  }
  &__item {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
    border-right: 1px solid $border;
    border-top: 1px solid $border;
  }
  .title {
    padding: 5px 0;
    background: $bgGray;
    min-width: 70px;
    border-top: none;
    border-bottom: 1px solid $border;
  }
}
.grow-normal {
  flex-grow: 1;
}
.grow-large {
  flex-grow: 2;
}
.grow-extra {
  flex-grow: 3;
}
.align-center {
  align-items: center;
}
.justify-center {
  justify-content: center;
}
.size-1 {
  height: 46px;
}
.size-2 {
  height: 23px;
}
.layer-title {
  flex: 0 0 47px;
}
</style>
