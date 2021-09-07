<template>
  <div class="board">
    <scrollbar>
      <div class="wrapper">
        <at-collapse :value="collapse">
          <at-collapse-item class="mt-3" title="Лоссы" center></at-collapse-item>
          <at-collapse-item class="mt-3" title="Метрики" center></at-collapse-item>
          <at-collapse-item class="mt-3" title="Промежуточные результаты" center></at-collapse-item>
          <at-collapse-item class="mt-3" title="Прогресс обучения" center></at-collapse-item>
          <at-collapse-item class="mt-3" title="Таблица прогресса обучения" center></at-collapse-item>
          <at-collapse-item class="mt-3" title="Статистические данные" center></at-collapse-item>
          <at-collapse-item class="mt-3" title="Баланс данных" center></at-collapse-item>
        </at-collapse>
      </div>
    </scrollbar>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
// import Images from "./main/images/index.vue";
// import Texts from "./main/texts/index.vue";
// import Scatters from "./main/Scatters.vue";
// import Chars from "./main/chars/index.vue";

export default {
  name: 'Graphics',
  components: {
    // Images,
    // Texts,
    // Scatters,
    // Chars,
  },
  computed: {
    ...mapGetters({
      chars: 'trainings/getToolbarChars',
      scatters: 'trainings/getToolbarScatters',
      images: 'trainings/getToolbarImages',
      texts: 'trainings/getToolbarTexts',
      // height: "settings/autoHeight",
    }),
  },
};
</script>

<style scoped>
.wrapper {
  padding: 20px;
  display: flex;
  flex-direction: column;
  flex-wrap: wrap;
}
</style>