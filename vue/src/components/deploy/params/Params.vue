<template>
  <div class="params">
      <div class="params-container__name">Загрузка в демо-панель</div>
      <div class="params-container pa-5">
        <div class="label">Название папки</div>
        <div class="t-input">
          <label class="t-input__label">https://demo.neural-university.ru/login/project/{{ deploy }}</label>
          <input v-model="deploy" class="t-input__input" type="text" name="deploy[deploy]"  @blur="$emit('blur', $event.target.value)" />
        </div>
        <Checkbox :label="'Перезаписать с таким же названием папки'" :type="'checkbox'" class="pd__top" @change="UseReplace"/>
        <Checkbox :label="'Использовать пароль для просмотра страницы'" :type="'checkbox'" @change="UseSec"/>
        <div class="password" v-if="use_sec">
          <div class="t-input">
            <input :type="passwordShow ? 'text' : 'password' " placeholder="Введите пароль" v-model="sec">
            <div class="password__icon">
              <i
                :class="['t-icon', passwordShow ? 'icon-deploy-password-open' : 'icon-deploy-password-close']"
                :title="'show password'"
                @click="passwordShow = !passwordShow">
              </i>
            </div>
          </div>
          <div class="t-input">
            <input :type="passwordShow ? 'text' : 'password' " placeholder="Подтверждение пароля" v-model="sec_accept">
            <div class="password__icon">
              <i :class="['t-icon', checkCorrect]"
                 :title="'is correct'">
              </i>
            </div>
          </div>
        </div>
        <button :disabled="send_disabled || !dataLoaded" @click="SendData" v-if="!DataSent">Отправить</button>
        <div class="loader" v-if="DataLoading">
          <div class="loader__title">Дождитесь окончания загрузки</div>
          <div class="loader__time">
            Загружено 892 MB из 1.2 GB    Осталось: меньше минуты
          </div>
          <div class="loader__progress">
            <div class="progress-bar">
              <div class="loading">
                <span>78%</span>
              </div>
            </div>
          </div>
        </div>
        <div class="req-ans" v-if="DataSent">
          <div class="answer__success">Загрузка завершена!</div>
          <div class="answer__label">Ссылка на сформированную загрузку</div>
          <div class="answer__url"><i :class="['t-icon', 'icon-deploy-copy']" :title="'copy'" @click="Copy(moduleList.url)"></i><span>{{ moduleList.url }}</span></div>
        </div>
        <ModuleList v-if="DataSent" :moduleList="moduleList.api_text"/>

      </div>
  </div>
</template>

<script>
import { mapGetters } from 'vuex'
import Checkbox from "@/components/forms/Checkbox";
import ModuleList from "./ModuleList";
import progress from "../../../at-ui/src/components/progress/src/progress";
export default {
  name: "Settings",
  components: {
    Checkbox,
    ModuleList
  },
  data: () => ({
    deploy: "",
    replace: false,
    use_sec: false,
    sec: "",
    sec_accept: "",
    send_disabled: true,
    DataSent: false,
    DataLoading: false,
    passwordShow: false
  }),
  computed: {
  ...mapGetters({
    height: "settings/height",
    moduleList: "deploy/getModuleList",
    dataLoaded: "deploy/getDataLoaded",
  }),
    checkCorrect(){
      return this.sec == this.sec_accept ? 'icon-deploy-password-correct' : 'icon-deploy-password-incorrect'
    },
  },
  mounted() {
    // console.log(this.moduleList)
  },
  watch: {
    deploy(val){
      if(val !== '') this.send_disabled = false
      else this.send_disabled = true
    },
    sec_accept(val){
      if(this.use_sec){
        if(val == this.sec) this.send_disabled = false;
        else this.send_disabled = true;
      }
    }
  },
  methods: {
    click() {
      console.log();
    },
    Percents(number){
      let loading = document.querySelector(".progress-bar > .loading");
      loading.style.width = number + "%";
      loading.find("span").value = number;
    },
    Copy(text){
      var textArea = document.createElement("textarea");
      textArea.value = text;

      textArea.style.top = "0";
      textArea.style.left = "0";
      textArea.style.position = "fixed";

      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();

      try {
        document.execCommand('copy');
      } catch (err) {
        console.error('Fallback: Oops, unable to copy', err);
      }

      document.body.removeChild(textArea);
    },
    UseSec(data) {
      this.use_sec = data.value;
    },
    UseReplace(data) {
      this.replace = data.value;
    },
    async progress(){
        let answer = await this.$store.dispatch('deploy/CheckProgress');
        if(!answer.finished){
          this.Percents(answer.percent);
          progress();
        }
    },
    async SendData(){
      let data = {
        deploy: this.deploy,
        replace: this.replace,
        use_sec: this.use_sec,
      }
      if(this.use_sec) data['sec'] = this.sec
      this.DataLoading = true;
      this.progress();
      await this.$store.dispatch('deploy/SendDeploy', data);
      this.DataLoading = false;
      this.DataSent = true;
    }
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
.t-input{
  padding-bottom: 10px;
}
.loader__title{
  font-size: 14px;
  line-height: 24px;
}
.loader__time{
  padding-top: 20px;
  font-size: 12px;
  line-height: 24px;
  color: #A7BED3;
  width: 100%;
  text-align: center;
}
.loader__progress{
  padding-top: 10px;
  width: 100%;
  display: flex;
  justify-content: center;
}
.progress-bar{
  width: 426px;
  background: #2B5278;
  border-radius: 4px;
}
.loading{
  background: #65B9F4;
  border-radius: 4px;
  width: 78%;
  span{
    position: relative;
    margin-left: 201px;
  }
}
.t-input{
  &__label{
    color: #6C7883;
    display: block;
    margin: 0 0 10px 0;
    line-height: 1.25;
    font-size: .75rem;
    user-select: none;
  }
  &__input{
    color: #fff;
    border-color: #6C7883;
    background: #242F3D;
    width: 100%;
  }
}
.req-ans{
  .answer__success{
    font-size: 14px;
    line-height: 24px;
    color: #FFFFFF;
    padding-top: 10px;
  }
  .answer__label{
    color: #A7BED3;
    font-size: 12px;
    line-height: 24px;
    padding-top: 15px;
  }
  .answer__url{
    font-size: 14px;
    line-height: 24px;
    color: #65B9F4;
    display: flex;
    span{
      padding-left: 10px;
    }
    i{
      cursor: pointer;
      width: 32px;
    }
  }
}
.password__icon{
  position: absolute;
  width: 415px;
  i{
    position: relative;
    float: right;
    margin-top: -34px;
  }
}

</style>