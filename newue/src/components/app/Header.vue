<template>
  <div class="header">
    <div class="header-left">
      <div href="#" class="header-left--logo"  @click="isShowNavigation = !isShowNavigation"></div>
      <div class="header-dropdown" v-if="isShowNavigation">
        <ul>
            <li
            v-for="route in routes"
              class="header-dropdown__item"
              :key="route.id"
              @click.prevent="selectedGeneratedId = route.id"
            >
              {{ route.title }}
            </li>
          <li class="header-dropdown__item header-dropdown__item--border">Тестовое поле</li>
        </ul>
      </div>
      <svg  @click="isShowNavigation = !isShowNavigation" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 14.5L17 9.5H7L12 14.5Z" fill="#6C7883" />
      </svg>
      <div class="header-left__menu">
        <ul>
          <li class="header-left__menu-item" v-for="(select, i) in selectedGenerated" :key="select.title + i">
            {{  select.title }}
          </li>
        </ul>
      </div>
    </div>
    <div class="header-right">
      <div class="header-right__line"></div>
      <div v-for="({ title, icon }, i) of iconRight" :key="'menu_' + i" class="header-right__icon" :title="title">
        <i :class="[icon]"></i>
      </div>
      <div @click="DialogProfile = true">
        <div class="header-right__icon">
          <i class="profile"></i>
        </div>
      </div>
    </div>
    <DModal style="position: absolute;" v-model="DialogProfile" title="Мой профиль">
      <t-field class="profile-modal-content__wrapper" label="Имя *">
        <DInputText v-model.trim="firstName" :error="errFirst" @input="errFirst=''"/>
      </t-field>
      <t-field class="profile-modal-content__wrapper" label="Фамилия *">
        <DInputText v-model.trim="lastName" :error="errLast" @input="errLast=''"/>
      </t-field>
       <t-field class="profile-modal-content__wrapper" label="Логин">
        <p>{{ user.login }}</p>
      </t-field>
      <t-field class="profile-modal-content__wrapper" label="E-mail">
        <p>{{ user.email }}</p>
      </t-field>
      <t-field class="profile-modal-content__wrapper" label="Token">
        <p>
          <span ref="token">{{ user.token }}</span>
          <svg @click="copy" class="btn-copy" width="21" height="21" viewBox="0 0 21 21" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12.9999 20.9999H2.99995C2.46402 21.0186 1.94442 20.8138 1.56524 20.4347C1.18605 20.0555 0.981324 19.5359 0.999948 18.9999V8.99995C0.981324 8.46402 1.18605 7.94442 1.56524 7.56524C1.94442 7.18605 2.46402 6.98132 2.99995 6.99995H6.99995V2.99995C6.98132 2.46402 7.18605 1.94442 7.56524 1.56524C7.94442 1.18605 8.46402 0.981324 8.99995 0.999948H18.9999C19.5359 0.981324 20.0555 1.18605 20.4347 1.56524C20.8138 1.94442 21.0186 2.46402 20.9999 2.99995V12.9999C21.0183 13.5358 20.8134 14.0552 20.4343 14.4343C20.0552 14.8134 19.5358 15.0183 18.9999 14.9999H14.9999V18.9999C15.0183 19.5358 14.8134 20.0552 14.4343 20.4343C14.0552 20.8134 13.5358 21.0183 12.9999 20.9999ZM2.99995 8.99995V18.9999H12.9999V14.9999H8.99995C8.46411 15.0183 7.94468 14.8134 7.56557 14.4343C7.18645 14.0552 6.98162 13.5358 6.99995 12.9999V8.99995H2.99995ZM8.99995 2.99995V12.9999H18.9999V2.99995H8.99995Z"/>
          </svg>
        </p>
        <p class="copy-msg"></p>
        <div @click="updateTokenProfile" class="btn-text">Обновить токен</div>
      </t-field>
      <t-field class="profile-modal-content__wrapper" v-show="showNoticeProfile">
        <div>
          <i class="notice__icon"></i>
          <p>{{ noticeMsgProfile }}</p>
        </div>
      </t-field>
      <template slot="footer">
        <DButton style="margin-left: auto;" @click="saveProfile" :loading="isLoadingProfile" :disabled="isLoadingProfile" color="primary" direction="left" >Сохранить</DButton>
      </template>
    </DModal>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';

export default {
  name: 'app-header',
  components:{
    DModal: () => import('@/components/global/modals/DModal'),
    DButton: () => import('@/components/global/forms/components/DButton'),
    DInputText: () => import('@/components/global/forms/components/DInputText'),
  },
  data: () => ({
    DialogProfile: false,
    isChangedProfile: false,
    showNoticeProfile: false,
    noticeMsgProfile: '',
    errFirst: '',
    errLast: '',
    isLoadingProfile: false,


    iconRight: [
      {
        title: 'вопрос',
        icon: 'icon-project-ask',
      },
      {
        title: 'уведомление',
        icon: 'icon-project-notification',
      },
    ],
    isShowNavigation: false,
    selectedGeneratedId: 1,
  }),
  computed: {
    ...mapGetters({
      user: 'projects/getUser',
    }),
    firstName: {
      set(first_name) {
        this.$store.commit('projects/SET_USER', { first_name });
      },
      get() {
        return this.user.first_name;
      },
    },
    lastName: {
      set(last_name) {
        this.$store.commit('projects/SET_USER', { last_name });
      },
      get() {
        return this.user.last_name;
      },
    },
    routes() {
      return [
        {
          id: 1,
          title: "Данные",
        },
        {
          id: 2,
          title: "Проектирование",
        },
        {
          id: 3,
          title: "Обучение",
        },
        {
          id: 4,
          title: "Проекты",
        }
      ]
    },
    generated(){
      return {
        1: [
          {
            title: 'Датасеты' 
          },
          {
            title:'Создание датасета'
          },
          {
            title:'Разметка'
          },
          {
            title:'Просмотр датасета'
          }
        ]
      }
    },
    selectedGenerated(){
      return  this.generated[this.selectedGeneratedId]
    },
  },
  methods:{
    copy() {
      let selection = window.getSelection();
      let range = document.createRange();

      range.selectNodeContents(this.$refs.token);
      selection.removeAllRanges();
      selection.addRange(range);
      document.execCommand('copy');
      selection.removeAllRanges();
      document.querySelector('.copy-msg').textContent = 'Token скопирован в буфер обмена'
    },
    async updateTokenProfile() {
      const res = await this.$store.dispatch('axios', { url: '/profile/update_token/' });
      if (res.success) {
        this.$refs.token.textContent = res.data.new_token;
        this.$store.dispatch('messages/setMessage', { message: `Ваш token успешно обновлен` });
      }
    },
    async saveProfile() {
      if (!this.firstName) this.errFirst = 'Поле обязательно для заполнения';
      if (!this.lastName) this.errLast = 'Поле обязательно для заполнения';

      if (this.firstName && this.lastName) {
        this.isLoadingProfile = true;
        const res = await this.$store.dispatch('profile/save', {
          first_name: this.firstName,
          last_name: this.lastName,
        });
        this.isLoadingProfile = false;

        if (res.success) this.$store.dispatch('messages/setMessage', { message: `Ваши данные успешно изменены` });
      }
    },
  },
  watch: {
    firstName() {
      this.isChangedProfile = true;
    },
    lastName() {
      this.isChangedProfile = true;
    },
  }
};
</script>

<style lang="scss" scoped>

// Modal

.profile-modal{
  &-content{
    &__wrapper{
      margin-bottom: 25px;
        .btn-text {
        display: inline-block;
        margin-top: 10px;
        cursor: pointer;
        color: #65b9f4;
        font-size: 14px;
        line-height: 24px;
        height: 27px;
        &:hover {
          border-bottom: 1px solid #65b9f4;
        }
      }

      .btn-copy {
        position: absolute;
        top: -10px;
        right: 0;
        cursor: pointer;
        width: 21px;
        height: 21px;
        transition: fill .3s ease;
        fill: #fff;
        &:hover{
          fill: #65b9f4;
        }
      }

      .copy-msg {
        font-size: 13px;
        margin-top: -2px;
        color: #3eba31;
      }
    }
  }
}

.header {
  background: #17212b;
  width: 100%;
  height: 52px;
  padding: 0 10px;
  position: fixed;
  left: 0;
  top: 0;
  z-index: 800;
  display: flex;
  flex-direction: row;
  flex-wrap: nowrap;
  justify-content: center;
  align-content: center;
  align-items: center;
  &-dropdown {
    position: absolute;
    top: 40px;
    background: #242f3d;
    box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.25);
    border-radius: 4px;
    &__item {
      padding: 8px 20px;
      color: #a7bed3;
      font-size: 14px;
      cursor: pointer;
      transition: color 0.3s ease, background 0.3s ease;
      &:first-child {
        padding-top: 12px;
      }
      &:last-child {
        padding-bottom: 12px;
      }
      &:hover {
        color: #65b9f4;
        background: #1e2734;
      }

      &--border {
        border-top: 1px solid #6c7883;
      }
    }
  }
  &-left {
    position: relative;
    display: flex;
    flex-direction: row;
    flex-wrap: nowrap;
    justify-content: flex-start;
    align-content: flex-start;
    align-items: center;
    &__menu{
      ul{
        display: flex;
      }
      &-item{
        margin-left: 40px;
        color: #A7BED3;
        font-size: 14px;
        cursor: pointer;
      }
    }
    svg {
      cursor: pointer;
    }
    &--logo {
      display: block;
      cursor: pointer;
      content: '';
      width: 28px;
      height: 28px;
      background-position: center;
      background-repeat: no-repeat;
      background-size: contain;
      background-image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjgiIGhlaWdodD0iMjgiIHZpZXdCb3g9IjAgMCAyOCAyOCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTIxLjQxNzUgOC4wNzA2QzIyLjIzMjggOC4wNzA2IDIyLjg5OTkgNy40MDM1NCAyMi44OTk5IDYuNTg4MjVDMjIuODk5OSA1Ljc3Mjk2IDIyLjIzMjggNS4xMDU5IDIxLjQxNzUgNS4xMDU5SDYuNTk0MDJDNS43Nzg3MiA1LjEwNTkgNS4xMTE2NiA1Ljc3Mjk2IDUuMTExNjYgNi41ODgyNUM1LjExMTY2IDcuNDAzNTQgNS43Nzg3MiA4LjA3MDYgNi41OTQwMiA4LjA3MDZIMjEuNDE3NVpNMTIuNTIzNCAyMS40MTE4QzEyLjUyMzQgMjIuMjI3MSAxMy4xOTA1IDIyLjg5NDEgMTQuMDA1OCAyMi44OTQxQzE0LjgyMTEgMjIuODk0MSAxNS40ODgxIDIyLjIyNzEgMTUuNDg4MSAyMS40MTE4VjEyLjc2NDdDMTUuNDg4MSAxMS45NDk0IDE0LjgyMTEgMTEuMjgyNCAxNC4wMDU4IDExLjI4MjRDMTMuMTkwNSAxMS4yODI0IDEyLjUyMzQgMTEuOTQ5NCAxMi41MjM0IDEyLjc2NDdWMjEuNDExOFoiIGZpbGw9IiM2NUI5RjQiLz4KPHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0yNy4xNzY1IDAuODIzNTI5SDAuODIzNTI5VjI3LjE3NjVIMjcuMTc2NVYwLjgyMzUyOVpNMCAwVjI4SDI4VjBIMFoiIGZpbGw9IiM2NUI5RjQiLz4KPC9zdmc+Cg==);
    }
    &--title {
      margin: 0 0 0 15px;
      line-height: 1.375;
      display: flex;
    }
    &--label {
      color: #a7bed3;
      margin: 0 5px 0 0;
      user-select: none;
    }
    &--name {
      position: relative;
      white-space: nowrap;
      font-weight: 700;
      display: flex;
      align-items: center;
      > span {
        min-width: 20px;
        height: 100%;
        margin: 0 7px 0 0;
        cursor: text;
        user-select: none;
        white-space: nowrap;
        outline: none;
        border: 1px solid transparent;
        border-radius: 4px;
      }
      > i {
        display: block;
        width: 13px;
        height: 13px;
        background-position: center;
        background-repeat: no-repeat;
        background-size: contain;
        background-image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTMiIGhlaWdodD0iMTMiIHZpZXdCb3g9IjAgMCAxMyAxMyIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTAuODc1IDkuNzgyODFWMTIuMTI2NkgzLjMxNjQxTDEwLjUxNjkgNS4yMTQwNkw4LjA3NTUyIDIuODcwMzFMMC44NzUgOS43ODI4MVpNMTIuNDA0OSAzLjQwMTU2QzEyLjY1ODkgMy4xNTc4MSAxMi42NTg5IDIuNzY0MDYgMTIuNDA0OSAyLjUyMDMxTDEwLjg4MTUgMS4wNTc4MUMxMC42Mjc2IDAuODE0MDYyIDEwLjIxNzQgMC44MTQwNjIgOS45NjM1NCAxLjA1NzgxTDguNzcyMTMgMi4yMDE1NkwxMS4yMTM1IDQuNTQ1MzFMMTIuNDA0OSAzLjQwMTU2WiIgZmlsbD0iI0E3QkVEMyIvPgo8L3N2Zz4K);
      }
    }
  }
  &__center {
    font-size: 1rem;
    font-weight: 600;
    text-align: center;
    white-space: nowrap;
  }
  &-right {
    display: flex;
    flex-direction: row;
    flex-wrap: nowrap;
    justify-content: flex-end;
    align-content: center;
    align-items: center;
    &__line {
      width: 1px;
      height: 30px;
      background: #242f3d;
      margin-right: 20px;
    }
    &__icon {
      margin: 0 10px 0 0;
      > i {
        background-size: 24px 24px;
        display: block;
        width: 32px;
        height: 32px;
        background-position: center;
        background-repeat: no-repeat;
        cursor: pointer;
        user-select: none;
      }
      .profile {
        background-size: 32px 32px;
        background-image: url('data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiA/PjxzdmcgaGVpZ2h0PSI0OCIgdmlld0JveD0iMCAwIDQ4IDQ4IiB3aWR0aD0iNDgiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZmlsbD0iIzY1QjlGNCIgZD0iTTI0IDRjLTExLjA1IDAtMjAgOC45NS0yMCAyMHM4Ljk1IDIwIDIwIDIwIDIwLTguOTUgMjAtMjAtOC45NS0yMC0yMC0yMHptMCA2YzMuMzEgMCA2IDIuNjkgNiA2IDAgMy4zMi0yLjY5IDYtNiA2cy02LTIuNjgtNi02YzAtMy4zMSAyLjY5LTYgNi02em0wIDI4LjRjLTUuMDEgMC05LjQxLTIuNTYtMTItNi40NC4wNS0zLjk3IDguMDEtNi4xNiAxMi02LjE2czExLjk0IDIuMTkgMTIgNi4xNmMtMi41OSAzLjg4LTYuOTkgNi40NC0xMiA2LjQ0eiIvPjxwYXRoIGQ9Ik0wIDBoNDh2NDhoLTQ4eiIgZmlsbD0ibm9uZSIvPjwvc3ZnPg==');
      }
    }
  }
  & > * {
    order: 0;
    flex: 1 1 auto;
    align-self: auto;
  }
}

.flexbox-center-nowrap {
  padding: 0 10px;
  margin: 1px 0 0 0;
}
.value {
  padding: 0;
  max-width: 300px;
}
.left--name_span[contenteditable='true'] {
  border: 1px solid #ffffff;
}
</style>
