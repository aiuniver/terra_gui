<template>
  <div class="board">
    <scrollbar>
      <div class="wrapper">
        <at-collapse @on-change="change">
          <at-collapse-item class="mt-3" title="Лоссы" center>
            <LoadSpiner v-show="loading" />
            <LossGraphs v-if="show" @isLoad="loading = false" />
          </at-collapse-item>
          <at-collapse-item class="mt-3" title="Метрики" center>
            <MetricGraphs @isLoad="loading = false" />
          </at-collapse-item>
          <at-collapse-item class="mt-3" title="Метрики" center></at-collapse-item>
          <at-collapse-item class="mt-3" title="Промежуточные результаты" center>
            <PrePesults/>
            <!-- <Images /> -->
            <Prediction />
          </at-collapse-item>
          <at-collapse-item class="mt-3" title="Прогресс обучения" center>
            <Progress />
          </at-collapse-item>
          <at-collapse-item class="mt-3" title="Таблица прогресса обучения" center>
            <Texts />
          </at-collapse-item>
          <at-collapse-item class="mt-3" title="Статистические данные" center>
            <Stats />
          </at-collapse-item>
          <at-collapse-item class="mt-3" title="Баланс данных" center>
            <Balance />
          </at-collapse-item>
        </at-collapse>
      </div>
    </scrollbar>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
// import Images from './main/images/index.vue';
// import Images from "./main/images/index.vue";
import Texts from './main/texts/index.vue';
import Progress from './main/progress/';
import LoadSpiner from '../forms/LoadSpiner.vue';
import Stats from './main/stats';
import Balance from './main/balance';
import Prediction from './main/prediction';

export default {
  name: 'Graphics',
  components: {
    Prediction,
    Texts,
    // Images,
    Progress,
    LoadSpiner,
    Stats,
    Balance,
    LossGraphs: () => import('./main/chars/LossGraphs.vue'),
    MetricGraphs: () => import('./main/chars/MetricGraphs.vue'),
  },
  data: () => ({
    collabse: [],
    loading: true,
  }),
  computed: {
    ...mapGetters({
      chars: 'trainings/getToolbarChars',
      scatters: 'trainings/getToolbarScatters',
      images: 'trainings/getToolbarImages',
      texts: 'trainings/getToolbarTexts',
      // height: "settings/autoHeight",
    }),
    show() {
      return this.collabse.includes('0');
    },
  },
  methods: {
    change(e) {
      this.$emit('collabse', this.collabse);
      this.collabse = e;
      console.log(e);
    },
  },
};
</script>

<style scoped>
.board {
  flex: 1 1 auto;
}
.wrapper {
  padding: 20px;
  display: flex;
  flex-direction: column;
  flex-wrap: wrap;
}
</style>