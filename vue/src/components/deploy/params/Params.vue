<template>
  <div class="params">
    <scrollbar :style="height">
      <div class="params-container__name">Загрузка в демо-панель</div>
      <div class="params-container pa-5">
        <div class="label">Название папки</div>
        <t-input :label="'https://demo.neural-university.ru/login/project/'" :labelColor="'#6C7883'"/>
        <Checkbox :label="'Перезаписать с таким же названием папки'" :type="'checkbox'" class="pd__top"/>
        <Checkbox :label="'Использовать пароль для просмотра страницы'" :type="'checkbox'"/>
        <button>Отправить</button>
      </div>
    </scrollbar>
  </div>
</template>

<script>
import { mapGetters } from 'vuex'
import TInput from "@/components/forms/TInput.vue";
import Checkbox from "@/components/forms/Checkbox";
export default {
  name: "Settings",
  components: {
    TInput,
    Checkbox
  },
  data: () => ({}),
  computed: {
  ...mapGetters({
    height: "settings/height",
  }),
  },
  methods: {
    click() {
      console.log();
    },
  },
};
</script>

<style lang="scss" scoped>
.params {
  flex-shrink: 0;
  width: 470px;
  border-left: #0e1621 solid 1px;
}
.params-container__name{
  padding: 30px 0 0 20px;
}
.pd__top{
  padding: 10px 0;
}
.label{
  color: #A7BED3;
  font-size: 12px;
  line-height: 24px;
}
.color__grey{
  color: #6C7883;
}
button {
  font-size: 0.875rem;
  margin-top: 10px;
  width: 107px;
}
</style>