<template>
  <div class="table" v-if="show">
    <scrollbar>
      <div class="table__columns">
        <div class="table__column">
          <div class="table__item title size-1 layer-title">Слой</div>
          <div class="table__item" v-for="(value, index) in Object.keys(this.predict).length" :key="index">
            {{ index }}
          </div>
        </div>

        <div
          class="table__column"
          v-if="Object.keys(this.predict).length > 0 && Object.keys(this.predict[1]['initial_data']).length > 0"
        >
          <div class="table__column">
            <div class="table__item title size-2">Исходные данные</div>
            <div class="table__row" v-for="(sloy, index) in predict" :key="'id_' + index">
              <div class="table__column" v-for="(value, input_name) in sloy.initial_data" :key="input_name">
                <div class="table__item title size-2" v-if="index == 1">{{ input_name }}</div>
                <div class="table__item">
                  <TableImage v-if="value.type == 'Image'" :image="value" />
                  <TableText v-if="value.type === 'str' || value.type === 'number'" :data="value" />
                  <TableTag v-if="value.type === 'Text'" :data="value" />

                  <Embed v-if="value.type === 'Video'" :src="value.data"></Embed>
                  <TableAudio v-if="value.type === 'Audio'" :url="value.data" />
                </div>
              </div>
            </div>
          </div>
        </div>

        <div
          class="table__column"
          v-if="Object.keys(this.predict).length > 0 && Object.keys(this.predict[1]['true_value']).length > 0"
        >
          <div class="table__column">
            <div class="table__item title size-2">Истинное значение</div>
            <div class="table__row" v-for="(sloy, index) in predict" :key="'id_' + index">
              <div class="table__column" v-for="(value, input_name) in sloy.true_value" :key="input_name">
                <div class="table__item title size-2" v-if="index == 1">{{ input_name }}</div>
                <div class="table__item">
                  <TableImage v-if="value.type == 'Image'" :image="value" />
                  <TableText v-if="value.type === 'str' || value.type === 'number'" :data="value" />
                  <TableTag v-if="value.type === 'Text'" :data="value" />

                  <Embed v-if="value.type === 'Video'" :src="value.data"></Embed>
                  <TableAudio v-if="value.type === 'Audio'" :url="value.data" />
                </div>
              </div>
            </div>
          </div>
        </div>

        <div
          class="table__column"
          v-if="Object.keys(this.predict).length > 0 && Object.keys(this.predict[1]['predict_value']).length > 0"
        >
          <div class="table__column">
            <div class="table__item title size-2">Предсказание</div>
            <div class="table__row" v-for="(sloy, index) in predict" :key="'id_' + index">
              <div class="table__column" v-for="(value, input_name) in sloy.predict_value" :key="input_name">
                <div class="table__item title size-2" v-if="index == 1">{{ input_name }}</div>
                <div class="table__item">
                  <TableImage v-if="value.type == 'Image'" :image="value" />
                  <TableText v-if="value.type === 'str' || value.type === 'number'" :data="value" />
                  <TableTag v-if="value.type === 'Text'" :data="value" />

                  <Embed v-if="value.type === 'Video'" :src="value.data"></Embed>
                  <TableAudio v-if="value.type === 'Audio'" :url="value.data" />
                </div>
              </div>
            </div>
          </div>
        </div>

        <div
          class="table__column"
          v-if="Object.keys(this.predict).length > 0 && Object.keys(this.predict[1]['statistic_values']).length > 0"
        >
          <div class="table__column">
            <div class="table__item title size-2">Статистика примеров</div>
            <div class="table__row" v-for="(sloy, index) in predict" :key="'id_' + index">
              <div class="table__column" v-for="(value, input_name) in sloy.statistic_values" :key="input_name">
                <div class="table__item title size-2" v-if="index == 1">{{ input_name }}</div>
                <div class="table__item" v-for="(input_val, key) in value" :key="key">
                  <TableStatisticText :data="input_val" :value="key" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </scrollbar>
  </div>
</template>

<script>
import TableImage from '@/components/training/main/prediction/components/TableImage.vue';
import TableText from '@/components/training/main/prediction/components/TableText.vue';
import TableStatisticText from '@/components/training/main/prediction/components/TableStatisticText.vue';
import TableTag from '@/components/training/main/prediction/components/TableTag.vue';
import TableAudio from '../audio/TableAudio';
import Embed from 'v-video-embed/src/embed';
export default {
  name: 'TextTableTest',
  components: {
    TableImage,
    TableStatisticText,
    TableAudio,
    TableText,
    TableTag,
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
  flex: 0 0 46px;
}
</style>
