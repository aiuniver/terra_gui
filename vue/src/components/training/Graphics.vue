<template>
  <div class="board">
    <scrollbar>
      <div class="wrapper">
        <at-collapse @on-change="change">
          <at-collapse-item class="mt-3" title="Лоссы" center>
            <LoadSpiner v-show="loading" />
            <Chars v-if="show" @isLoad="loading = false" />
          </at-collapse-item>
          <at-collapse-item class="mt-3" title="Метрики" center></at-collapse-item>
          <at-collapse-item class="mt-3" title="Промежуточные результаты" center>
            <Images />
          </at-collapse-item>
          <at-collapse-item class="mt-3" title="Прогресс обучения" center>
            <Scatters />
          </at-collapse-item>
          <at-collapse-item class="mt-3" title="Таблица прогресса обучения" center>
            <Texts />
          </at-collapse-item>
          <at-collapse-item class="mt-3" title="Статистические данные" center></at-collapse-item>
          <at-collapse-item class="mt-3" title="Баланс данных" center></at-collapse-item>
        </at-collapse>
      </div>
    </scrollbar>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
import Images from "./main/images/index.vue";
import Texts from "./main/texts/index.vue";
import Scatters from "./main/Scatters.vue";
import LoadSpiner from '../forms/LoadSpiner.vue'
export default {
  name: 'Graphics',
  components: {
    Images,
    Texts,
    Scatters,
    LoadSpiner,
    Chars: () => import('./main/chars/index.vue'),
  },
  data: () => ({
    collabse: [],
    loading: true
  }),
  computed: {
    ...mapGetters({
      chars: 'trainings/getToolbarChars',
      scatters: 'trainings/getToolbarScatters',
      images: 'trainings/getToolbarImages',
      texts: 'trainings/getToolbarTexts',
      // height: "settings/autoHeight",
    }),
    show () {
      return this.collabse.includes('0')
    }
  },
  methods: {
    change(e) {
      this.collabse = e
      console.log(e)
    }
  }
};
</script>

<style scoped>
.board {
  height: 85%;
}
.wrapper {
  padding: 20px;
  display: flex;
  flex-direction: column;
  flex-wrap: wrap;
}
</style>